#!/usr/bin/env python3
import os, re, csv, time, argparse, random
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump as joblib_dump

# ────────────────────────────── Logging ────────────────────────────────
def log(msg): print(msg, flush=True)
def human_time(s):
    m, s = divmod(int(max(0, s)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ────────────────────────────── Files / IDs ────────────────────────────
ID_FROM_FILENAME = re.compile(r"^(?P<id>.+?)(?:_original_SAE)?\.pt$")
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
    # POS: token before ':' if present; NEG: keep full token (contains '|')
    if '|' in seq_id:
        return seq_id
    return seq_id.split(':', 1)[0] if ':' in seq_id else seq_id

def build_embed_index(embed_dir: str):
    id_to_path = {}
    for p in list_pt(embed_dir):
        sid = filename_to_id(os.path.basename(p))
        id_to_path.setdefault(sid, p)
    return id_to_path

# ────────────────────────────── Dataset build ──────────────────────────
def build_design_matrix(entries, id2path, top_feats, log_every=5000):
    """
    entries: list of (seq_id, fold, label)
    id2path: {seq_id -> path_to_pt}
    top_feats: list[int] of feature indices to extract (max_top features)
    Returns:
      X: np.ndarray (n_samples, len(top_feats))
      y: np.ndarray (n_samples,)
      folds: np.ndarray (n_samples,) with fold id per sample
      stats dict
    """
    n = len(entries)
    d = len(top_feats)
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    folds = np.empty((n,), dtype=np.int64)

    bad_load = bad_shape = missing = 0
    start = time.time()
    for i, (sid, fold, label) in enumerate(entries):
        path = id2path.get(sid)
        if path is None:
            missing += 1
            X[i, :] = 0.0
            y[i] = label
            folds[i] = fold
            continue
        try:
            t = safe_load_tensor(path)     # (L, F)
            v = t.max(dim=0).values.numpy()
            X[i, :] = v[top_feats]
            y[i] = label
            folds[i] = fold
        except EOFError:
            bad_load += 1; X[i,:]=0.0; y[i]=label; folds[i]=fold
        except ValueError:
            bad_shape += 1; X[i,:]=0.0; y[i]=label; folds[i]=fold

        if (i+1) % log_every == 0 or (i+1) == n:
            el = time.time()-start
            rate = (i+1)/el if el>0 else 0.0
            log(f"  [build] {i+1:>7}/{n:<7} | elapsed {human_time(el)} | {rate:6.1f} samp/s")

    stats = dict(total=n, missing=missing, bad_load=bad_load, bad_shape=bad_shape)
    return X, y, folds, stats

# ────────────────────────────── Training ───────────────────────────────
def train_val_for_outer_fold(X, y, folds_vec, remain_folds, step, max_top, seed):
    """
    For feature counts N in [step, max_top] by step:
      loop inner validation folds v in remain_folds:
        train on (remain_folds - {v}); validate on v
      average accuracy across inner folds.
    Prints per-fold inner accuracies and a per-N summary.
    Returns: list of dicts with keys: num_features, validation
    """
    results = []
    for num_feat in range(step, max_top+1, step):
        accs = []
        # optional banner per N:
        print(f"  [outer] evaluating N={num_feat} features across folds {remain_folds}")
        for v in remain_folds:
            train_idx = (folds_vec != v)
            val_idx   = (folds_vec == v)
            if not np.any(val_idx) or not np.any(train_idx):
                continue

            pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=None, random_state=seed))
            ])
            pipe.fit(X[train_idx, :num_feat], y[train_idx])
            pred = pipe.predict(X[val_idx, :num_feat])
            acc  = accuracy_score(y[val_idx], pred)

            # --- ADDED: per-fold inner accuracy line
            print(f"      [inner] N={num_feat:4d}  val_fold={v}  acc={acc*100:5.2f}%")

            accs.append(acc)

        val = float(np.mean(accs))*100.0 if accs else 0.0

        # --- ADDED: per-N summary line
        print(f"    [summary] N={num_feat:4d}  mean_val={val:5.2f}%  over {len(accs)} folds")

        results.append({"num_features": num_feat, "validation": f"{val:.2f}"})
    return results

# ────────────────────────────── Main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Nested CV logistic regression using per-excluded-fold rankings and predefined folds.")
    ap.add_argument("--positives", required=True, help="Folder with positive .pt tensors")
    ap.add_argument("--negatives", required=True, help="Folder with negative .pt tensors")
    ap.add_argument("--ranking",   required=True, help="Folder with feature_ranking_excluded_fold_{k}.csv")
    ap.add_argument("--models_outdir", required=True, help="Where to save refit models (trained on all non-excluded folds).")
    ap.add_argument("--val_results_outdir", required=True, help="Where to save val_results_excluded_fold_{k}.csv")
    # (keeping your original flag spelling)
    ap.add_argument("--postive_folds", required=True, help="CSV with columns AC, cluster (0..9) for positives")
    ap.add_argument("--negative_folds", required=True, help="CSV with columns entry, fold (0..9) for negatives")
    ap.add_argument("--permitted_pos", required=True, help="Permitted POS FASTA")
    ap.add_argument("--permitted_neg", required=True, help="Permitted NEG FASTA")
    ap.add_argument("--max_top",   type=int, default=1000, help="Max # of ranked features to consider")
    ap.add_argument("--step",      type=int, default=5, help="Step size for feature counts")
    ap.add_argument("--seed",      type=int, default=9687254, help="Random seed")
    args = ap.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.models_outdir, exist_ok=True)
    os.makedirs(args.val_results_outdir, exist_ok=True)

    # Load permitted IDs
    pos_perm = set(fasta_ids(args.permitted_pos))
    neg_perm = set(fasta_ids(args.permitted_neg))
    log(f"[permit] POS={len(pos_perm):,}  NEG={len(neg_perm):,}")

    # Build embedding indices
    pos_id2path = build_embed_index(args.positives)
    neg_id2path = build_embed_index(args.negatives)

    # Map POS IDs to folds via GraphPart CSV
    dfp = pd.read_csv(args.postive_folds)  # sic: keeping flag name
    if not {"AC","cluster"} <= set(dfp.columns):
        raise ValueError("Positive folds CSV must contain columns: AC, cluster")
    dfp["cluster"] = pd.to_numeric(dfp["cluster"], errors="coerce").round().astype("Int64")
    pos_ac2fold = {str(ac): int(clu) for ac, clu in zip(dfp["AC"].astype(str), dfp["cluster"]) if pd.notna(clu)}

    # Map NEG IDs to folds via provided CSV
    dfn = pd.read_csv(args.negative_folds)
    if not {"entry","fold"} <= set(dfn.columns):
        raise ValueError("Negative folds CSV must contain columns: entry, fold")
    dfn["fold"] = pd.to_numeric(dfn["fold"], errors="coerce").astype("Int64")
    neg_id2fold = {str(e): int(f) for e, f in zip(dfn["entry"].astype(str), dfn["fold"]) if pd.notna(f)}

    # Partition by fold (and filter to permitted + have an embedding path)
    byfold_pos_ids = defaultdict(list)
    byfold_neg_ids = defaultdict(list)

    pos_missing_fold = 0
    for sid in pos_perm:
        ac = seqid_to_ac_key(sid)
        f = pos_ac2fold.get(ac)
        if f is None or f<0 or f>9:
            pos_missing_fold += 1
            continue
        if sid in pos_id2path:  # keep only those with embeddings
            byfold_pos_ids[f].append(sid)
    if pos_missing_fold:
        log(f"[warn] POS permitted but missing fold assignment: {pos_missing_fold:,} (ignored)")

    neg_missing_fold = 0
    for sid in neg_perm:
        f = neg_id2fold.get(sid)
        if f is None or f<0 or f>9:
            neg_missing_fold += 1
            continue
        if sid in neg_id2path:
            byfold_neg_ids[f].append(sid)
    if neg_missing_fold:
        log(f"[warn] NEG permitted but missing fold assignment: {neg_missing_fold:,} (ignored)")

    # Outer loop: exclude fold k
    for k in range(10):
        log("\n" + "="*72)
        log(f"[outer] Excluding fold {k}")
        rank_path = os.path.join(args.ranking, f"feature_ranking_excluded_fold_{k}.csv")
        if not os.path.exists(rank_path):
            raise FileNotFoundError(f"Ranking file not found for excluded fold {k}: {rank_path}")

        # Load ranking (rank 1 = best)
        rnk = pd.read_csv(rank_path)
        if not {"rank","feature","F1_score"} <= set(rnk.columns):
            raise ValueError(f"Ranking CSV missing required columns: {rank_path}")
        rnk = rnk.sort_values("rank", ascending=True).reset_index(drop=True)
        feat_list = rnk["feature"].astype(int).tolist()
        if not feat_list:
            log(f"[outer {k}] no features found in ranking; writing empty results.")
            out_csv = os.path.join(args.val_results_outdir, f"val_results_excluded_fold_{k}.csv")
            pd.DataFrame(columns=["num_features","validation"]).to_csv(out_csv, index=False)
            continue

        max_top = min(args.max_top, len(feat_list))
        top_feats = feat_list[:max_top]
        log(f"[outer {k}] using top {max_top} ranked features from {os.path.basename(rank_path)}")

        # Gather entries from remaining folds only
        remain_folds = [f for f in range(10) if f != k]
        pos_entries = [(sid, f, 1) for f in remain_folds for sid in byfold_pos_ids.get(f, [])]
        neg_entries = [(sid, f, 0) for f in remain_folds for sid in byfold_neg_ids.get(f, [])]
        entries = pos_entries + neg_entries
        log(f"[outer {k}] training pool (k excluded): POS={len(pos_entries):,}  NEG={len(neg_entries):,}")

        if len(entries) == 0:
            log(f"[outer {k}] Empty training pool; writing empty results.")
            out_csv = os.path.join(args.val_results_outdir, f"val_results_excluded_fold_{k}.csv")
            pd.DataFrame(columns=["num_features","validation"]).to_csv(out_csv, index=False)
            continue

        # Build design matrix for remaining folds (max_pool → slice top_feats)
        t0 = time.time()
        X_pos, y_pos, f_pos, s_pos = build_design_matrix(pos_entries, pos_id2path, top_feats)
        X_neg, y_neg, f_neg, s_neg = build_design_matrix(neg_entries, neg_id2path, top_feats)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])
        folds_vec = np.concatenate([f_pos, f_neg])

        log(f"[outer {k}] built X={X.shape}  y={y.shape} in {human_time(time.time()-t0)}")
        if s_pos["missing"] or s_neg["missing"]:
            log(f"           missing embeddings: POS={s_pos['missing']:,} NEG={s_neg['missing']:,}")
        if s_pos["bad_load"] or s_pos["bad_shape"] or s_neg["bad_load"] or s_neg["bad_shape"]:
            log(f"           bad files (load/shape): POS={s_pos['bad_load']:,}/{s_pos['bad_shape']:,} "
                f"NEG={s_neg['bad_load']:,}/{s_neg['bad_shape']:,}")

        # Inner CV across remain_folds
        log(f"[outer {k}] inner-CV across folds: {remain_folds}")
        results = train_val_for_outer_fold(
            X, y, folds_vec, remain_folds,
            step=args.step, max_top=max_top, seed=args.seed
        )

        # Save validation results
        out_csv = os.path.join(args.val_results_outdir, f"val_results_excluded_fold_{k}.csv")
        pd.DataFrame.from_records(results).to_csv(out_csv, index=False)
        log(f"[outer {k}] wrote validation results → {out_csv}")

        # Optional: refit on ALL remaining folds (no inner holdout) per N and save models
        log(f"[outer {k}] refitting per N on all non-excluded folds (models for later test use)")
        for rec in results:
            nfeat = int(rec["num_features"])
            pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=None, random_state=args.seed))
            ])
            pipe.fit(X[:, :nfeat], y)
            model_path = os.path.join(args.models_outdir, f"logreg_excl{k}_top{nfeat}.joblib")
            joblib_dump(pipe, model_path)
        log(f"[outer {k}] models saved under → {args.models_outdir}")

if __name__ == "__main__":
    main()
