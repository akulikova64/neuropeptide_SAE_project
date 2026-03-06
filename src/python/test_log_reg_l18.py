#!/usr/bin/env python3
"""
Evaluate saved logistic-regression models on their corresponding excluded test fold.

For each outer fold k (0..9):
  • Load ranking file: /Volumes/T7 Shield/layer_18_ranking_files/feature_ranking_excluded_fold_{k}.csv
  • Build test set = all permitted POS+NEG whose fold == k
  • For every saved model logreg_excl{k}_top{N}.joblib:
        - slice top-N features using the ranking (rank 1..N)
        - compute accuracy on the test set
  • Save CSV: /Volumes/T7 Shield/layer_18_classifiers/test_results/test_results_excluded_fold_{k}.csv
        columns: num_features,test_accuracy

Assumptions / conventions (matches your training script):
  - POS fold map from ../../data/combined_datasets_graphpart_mmseqs.csv (columns: AC, cluster[0..9] possibly as 0.0)
  - NEG fold map from ../../data/negative_dataset_10_folds.csv (columns: entry, fold[0..9])
  - Permitted FASTAs provide the test universe; IDs must exist and have embeddings
  - Embeddings are per-sequence tensors (L, F); we use max-over-positions → (F,)
  - Embedding filenames: <ID>_original_SAE.pt (or <ID>.pt); ID token must match FASTA header token

Run example:
  python eval_log_reg_l18_test.py \
    --positives "/Volumes/T7 Shield/layer_18_embeddings" \
    --negatives "/Volumes/T7 Shield/layer_18_negative_embeddings" \
    --ranking   "/Volumes/T7 Shield/layer_18_ranking_files" \
    --models_dir "/Volumes/T7 Shield/layer_18_classifiers/models" \
    --test_results_outdir "/Volumes/T7 Shield/layer_18_classifiers/test_results" \
    --postive_folds "../../data/combined_datasets_graphpart_mmseqs.csv" \
    --negative_folds "../../data/negative_dataset_10_folds.csv" \
    --permitted_pos "../../data/combined_l18_positive_final.fasta" \
    --permitted_neg "../../data/negative_l18_dataset_final.fasta"
"""

import os, re, csv, time, argparse
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load

# ────────────────────────────── Logging ────────────────────────────────
def log(msg): print(msg, flush=True)
def human_time(s):
    m, s = divmod(int(max(0, s)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ────────────────────────────── Files / IDs ────────────────────────────
ID_FROM_FILENAME = re.compile(r"^(?P<id>.+?)(?:_original_SAE)?\.pt$")
MODEL_NAME_RE    = re.compile(r"^logreg_excl(?P<fold>\d+)_top(?P<nfeat>\d+)\.joblib$")

def filename_to_id(fn):
    m = ID_FROM_FILENAME.match(fn)
    if m: return m.group("id")
    return fn[:-3] if fn.endswith(".pt") else fn

def list_pt(folder):
    return sorted(p for p in glob(os.path.join(folder, "*.pt"))
                  if not os.path.basename(p).startswith("._"))

def safe_load_tensor(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    t = torch.load(path, map_location="cpu")
    if t.ndim != 2:
        raise ValueError(f"bad shape {tuple(t.shape)}")
    return t

def fasta_ids(path):
    ids = []
    with open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids

def seqid_to_ac_key(seq_id: str) -> str:
    # POS: token before ':' if present; NEG: keep full token (usually contains '|')
    if '|' in seq_id:
        return seq_id
    return seq_id.split(':', 1)[0] if ':' in seq_id else seq_id

def build_embed_index(embed_dir: str):
    id_to_path = {}
    for p in list_pt(embed_dir):
        sid = filename_to_id(os.path.basename(p))
        id_to_path.setdefault(sid, p)
    return id_to_path

# ────────────────────────────── Data assembly ──────────────────────────
def build_test_matrix(entries, id2path, top_feats, log_every=5000):
    """
    entries: list of (seq_id, label)
    id2path: {seq_id -> path}
    top_feats: list[int] → slice columns in this order
    Returns:
      X: (n, len(top_feats)), y: (n,), stats dict
    """
    n = len(entries)
    d = len(top_feats)
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)

    missing = bad_load = bad_shape = 0
    start = time.time()
    for i, (sid, label) in enumerate(entries):
        path = id2path.get(sid)
        if path is None:
            missing += 1
            X[i, :] = 0.0
            y[i]    = label
        else:
            try:
                t = safe_load_tensor(path)     # (L, F)
                v = t.max(dim=0).values.numpy()
                X[i, :] = v[top_feats]
                y[i]    = label
            except EOFError:
                bad_load += 1; X[i,:]=0.0; y[i]=label
            except ValueError:
                bad_shape += 1; X[i,:]=0.0; y[i]=label

        if (i+1) % log_every == 0 or (i+1) == n:
            el = time.time()-start
            rate = (i+1)/el if el>0 else 0.0
            log(f"  [test-build] {i+1:>7}/{n:<7} | elapsed {human_time(el)} | {rate:6.1f} samp/s")

    stats = dict(total=n, missing=missing, bad_load=bad_load, bad_shape=bad_shape)
    return X, y, stats

# ────────────────────────────── Main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate saved LR models on excluded test folds.")
    ap.add_argument("--positives", required=True, help="Folder with positive .pt tensors")
    ap.add_argument("--negatives", required=True, help="Folder with negative .pt tensors")
    ap.add_argument("--ranking",   required=True, help="Folder with feature_ranking_excluded_fold_{k}.csv")
    ap.add_argument("--models_dir", required=True, help="Folder with logreg_excl{k}_top{N}.joblib models")
    ap.add_argument("--test_results_outdir", required=True, help="Where to save test_results_excluded_fold_{k}.csv")
    ap.add_argument("--postive_folds", required=True, help="CSV with columns AC, cluster (0..9) for positives")
    ap.add_argument("--negative_folds", required=True, help="CSV with columns entry, fold (0..9) for negatives")
    ap.add_argument("--permitted_pos", required=True, help="Permitted POS FASTA")
    ap.add_argument("--permitted_neg", required=True, help="Permitted NEG FASTA")
    args = ap.parse_args()

    os.makedirs(args.test_results_outdir, exist_ok=True)

    # Load permitted IDs
    pos_perm = set(fasta_ids(args.permitted_pos))
    neg_perm = set(fasta_ids(args.permitted_neg))
    log(f"[permit] POS={len(pos_perm):,}  NEG={len(neg_perm):,}")

    # Embedding indices
    pos_id2path = build_embed_index(args.positives)
    neg_id2path = build_embed_index(args.negatives)

    # Positive fold map
    dfp = pd.read_csv(args.postive_folds)
    if not {"AC","cluster"} <= set(dfp.columns):
        raise ValueError("Positive folds CSV must contain columns: AC, cluster")
    dfp["cluster"] = pd.to_numeric(dfp["cluster"], errors="coerce").round().astype("Int64")
    pos_ac2fold = {str(ac): int(clu) for ac, clu in zip(dfp["AC"].astype(str), dfp["cluster"]) if pd.notna(clu)}

    # Negative fold map
    dfn = pd.read_csv(args.negative_folds)
    if not {"entry","fold"} <= set(dfn.columns):
        raise ValueError("Negative folds CSV must contain columns: entry, fold")
    dfn["fold"] = pd.to_numeric(dfn["fold"], errors="coerce").astype("Int64")
    neg_id2fold = {str(e): int(f) for e, f in zip(dfn["entry"].astype(str), dfn["fold"]) if pd.notna(f)}

    # Group permitted IDs by test fold (k)
    test_ids_pos = defaultdict(list)
    for sid in pos_perm:
        ac = seqid_to_ac_key(sid)
        k  = pos_ac2fold.get(ac, None)
        if k is not None and 0 <= k <= 9:
            if sid in pos_id2path:
                test_ids_pos[k].append(sid)

    test_ids_neg = defaultdict(list)
    for sid in neg_perm:
        k = neg_id2fold.get(sid, None)
        if k is not None and 0 <= k <= 9:
            if sid in neg_id2path:
                test_ids_neg[k].append(sid)

    # Iterate outer folds k
    for k in range(10):
        log("\n" + "="*72)
        log(f"[test] Evaluating excluded fold {k}")

        # Ranking for fold k
        rank_path = os.path.join(args.ranking, f"feature_ranking_excluded_fold_{k}.csv")
        if not os.path.exists(rank_path):
            log(f"[test] SKIP fold {k}: ranking file missing → {rank_path}")
            continue
        rnk = pd.read_csv(rank_path)
        if not {"rank","feature","F1_score"} <= set(rnk.columns):
            log(f"[test] SKIP fold {k}: ranking CSV missing required columns → {rank_path}")
            continue
        rnk = rnk.sort_values("rank", ascending=True).reset_index(drop=True)
        feat_order = rnk["feature"].astype(int).tolist()
        if not feat_order:
            log(f"[test] SKIP fold {k}: empty ranking.")
            continue

        # Collect models for this fold k
        model_glob = os.path.join(args.models_dir, f"logreg_excl{k}_top*.joblib")
        model_paths = sorted(glob(model_glob))
        if not model_paths:
            log(f"[test] SKIP fold {k}: no models found matching {model_glob}")
            continue

        # Build the *test* entries (fold k only)
        pos_entries = [(sid, 1) for sid in test_ids_pos.get(k, [])]
        neg_entries = [(sid, 0) for sid in test_ids_neg.get(k, [])]
        test_entries = pos_entries + neg_entries
        log(f"[test {k}] POS test={len(pos_entries):,}  NEG test={len(neg_entries):,}  total={len(test_entries):,}")

        if len(test_entries) == 0:
            log(f"[test {k}] No test entries; writing empty result CSV.")
            out_csv = os.path.join(args.test_results_outdir, f"test_results_excluded_fold_{k}.csv")
            pd.DataFrame(columns=["num_features","test_accuracy"]).to_csv(out_csv, index=False)
            continue

        # We will (re)build the test matrix as N grows, but reuse earlier columns to avoid reloads.
        # Strategy: build once at max-N for this fold, then slice [:N] per model.
        # Determine max N present among models (and bounded by ranking length)
        nfeats_in_models = []
        for mp in model_paths:
            mname = os.path.basename(mp)
            m = MODEL_NAME_RE.match(mname)
            if m and int(m.group("fold")) == k:
                nfeats_in_models.append(int(m.group("nfeat")))
        if not nfeats_in_models:
            log(f"[test {k}] No parsable models for this fold; skipping.")
            continue
        maxN = min(max(nfeats_in_models), len(feat_order))
        top_feats_max = feat_order[:maxN]

        # Build design matrix once at maxN
        t0 = time.time()
        X_pos, y_pos, s_pos = build_test_matrix(pos_entries, pos_id2path, top_feats_max)
        X_neg, y_neg, s_neg = build_test_matrix(neg_entries, neg_id2path, top_feats_max)
        X_full = np.vstack([X_pos, X_neg])
        y_full = np.concatenate([y_pos, y_neg])
        log(f"[test {k}] built test matrix X={X_full.shape} in {human_time(time.time()-t0)}")
        if s_pos["missing"] or s_neg["missing"]:
            log(f"           missing embeddings: POS={s_pos['missing']:,}  NEG={s_neg['missing']:,}")
        if s_pos["bad_load"] or s_pos["bad_shape"] or s_neg["bad_load"] or s_neg["bad_shape"]:
            log(f"           bad files (load/shape): POS={s_pos['bad_load']:,}/{s_pos['bad_shape']:,} "
                f"NEG={s_neg['bad_load']:,}/{s_neg['bad_shape']:,}")

        # Evaluate each model, slicing X_full columns to its N
        rows = []
        for mp in model_paths:
            mname = os.path.basename(mp)
            m = MODEL_NAME_RE.match(mname)
            if not m: 
                continue
            fold_in_name = int(m.group("fold"))
            if fold_in_name != k:
                continue
            N = int(m.group("nfeat"))
            if N < 1:
                continue
            N = min(N, X_full.shape[1])  # just in case

            model = joblib_load(mp)
            y_pred = model.predict(X_full[:, :N])
            acc = (y_pred == y_full).mean() * 100.0
            rows.append({"num_features": N, "test_accuracy": f"{acc:.2f}"})
            log(f"  [test {k}] model={mname}  N={N:4d}  accuracy={acc:5.2f}%")

        # Save per-fold test results
        out_csv = os.path.join(args.test_results_outdir, f"test_results_excluded_fold_{k}.csv")
        pd.DataFrame.from_records(sorted(rows, key=lambda r: int(r["num_features"]))).to_csv(out_csv, index=False)
        log(f"[test {k}] wrote → {out_csv}")

if __name__ == "__main__":
    main()
