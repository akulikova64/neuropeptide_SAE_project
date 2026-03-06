#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load
from Bio import SeqIO

from interplm.sae.inference import load_sae_from_hf
from interplm.esm.embed import embed_single_sequence


# ---------- warnings ----------
# Silence transformers FutureWarning about clean_up_tokenization_spaces default
warnings.filterwarnings(
    "ignore",
    message=r".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

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
    v = features_2d.max(dim=0).values
    return v.detach().cpu().numpy().astype(np.float32)

def count_fasta_records(path: Path) -> int:
    n = 0
    with path.open("r") as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n

def load_already_done(out_csv: Path) -> set[str]:
    """
    Read existing output CSV and return set of Entry IDs already scored.
    Expects header with 'Entry' in first column.
    """
    done = set()
    if not out_csv.exists():
        return done
    try:
        with out_csv.open("r", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return done
            # find Entry column (default first col)
            try:
                entry_idx = header.index("Entry")
            except ValueError:
                entry_idx = 0
            for row in reader:
                if not row:
                    continue
                if entry_idx < len(row):
                    done.add(row[entry_idx])
    except Exception:
        # If file is malformed, don't assume anything is done.
        return set()
    return done


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Stream FASTA → ESM embedding → SAE → LR prediction (append-only, resumable; no saving embeddings)."
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

    total = count_fasta_records(fasta_path)
    log(f"[scan] FASTA entries: {total:,}")

    # Resume support
    done = load_already_done(out_path)
    if done:
        log(f"[resume] Found {len(done):,} already-scored entries in {out_path} — will skip them.")

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
    already = 0
    start = time.time()

    # Open output in append mode; write header only if new/empty
    file_exists = out_path.exists()
    write_header = (not file_exists) or (out_path.stat().st_size == 0)

    with out_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["Entry", "probability"])
            fh.flush()

        for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta"), start=1):
            entry = record.id
            if entry in done:
                already += 1
                if (i % args.log_every) == 0:
                    log(f"  [skip] {i:,}/{total:,} (already done) — last: {entry}")
                continue

            seq = str(record.seq)

            # show current entry
            log(f"[{i}/{total}] {entry}")

            try:
                with torch.no_grad():
                    embedding = embed_single_sequence(
                        sequence=seq,
                        model_name=args.esm_model_name,
                        layer=args.plm_layer
                    )
                    features = sae.encode(embedding)
                    v = max_pool_features(features)

                x = v[feats].reshape(1, -1)
                proba = float(pipe.predict_proba(x)[0, 1])

                writer.writerow([entry, f"{proba:.6f}"])
                fh.flush()  # ensure it’s on disk right away
                done.add(entry)
                written += 1

            except Exception as e:
                skipped += 1
                log(f"  ⚠️ skip {entry} ({e})")

            # free memory
            for obj_name in ("embedding", "features", "v"):
                if obj_name in locals():
                    try:
                        del locals()[obj_name]
                    except Exception:
                        pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i % args.log_every) == 0:
                el = time.time() - start
                rate = (written + skipped) / el if el > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else 0.0
                log(f"  [prog] {i:,}/{total:,} | wrote={written:,} skipped={skipped:,} already={already:,} | "
                    f"elapsed {human_time(el)} | ETA {human_time(eta)} | rate {rate:.2f}/s")

    log(f"[done] CSV → {out_path}")
    log(f"[stats] wrote={written:,}, skipped={skipped:,}, already={already:,}")
    log(f"[time] total {human_time(time.time() - t0)}")


if __name__ == "__main__":
    main()