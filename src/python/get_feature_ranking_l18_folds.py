#!/usr/bin/env python3
"""
Per-excluded-fold feature ranking (nested-CV outer folds), PLUS:
- Create 10 equal folds (0..9) for the permitted NEGATIVE set with a fixed seed,
  and save as ../../data/negative_dataset_10_folds.csv

Outputs per fold k:
  /Volumes/T7 Shield/layer_18_ranking_files/winning_thresholds_excluded_fold_{k}.csv
    (best threshold per feature; columns: feature, winning_threshold, winning_F1_score)
  /Volumes/T7 Shield/layer_18_ranking_files/feature_ranking_excluded_fold_{k}.csv
    (ACTUAL RANKING; columns: rank, feature, F1_score)

Also writes (only if missing embeddings exist):
  overall_*_missing_embedding_ids.txt
  fold_{k}_*_missing_embedding_ids.txt

And:
  ../../data/negative_dataset_10_folds.csv
"""

import os
import re
import csv
import time
import random
from glob import glob
import pandas as pd
import torch

# ── Config (paths) ─────────────────────────────────────────────────────
POS_EMB_DIR = "/Volumes/T7 Shield/layer_18_embeddings/"
NEG_EMB_DIR = "/Volumes/T7 Shield/layer_18_negative_embeddings/"
POS_FASTA   = "../../data/combined_l18_positive_final.fasta"
NEG_FASTA   = "../../data/negative_l18_dataset_final.fasta"
FOLDS_CSV   = "../../data/combined_datasets_graphpart_mmseqs.csv"
NEG_FOLDS_OUT = "../../data/negative_dataset_10_folds.csv"
OUT_DIR     = "/Volumes/T7 Shield/layer_18_ranking_files/"
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds to sweep (strict >)
THRESHOLDS = torch.tensor([0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=torch.float32)

# ── Helpers ────────────────────────────────────────────────────────────
ID_FROM_FILENAME = re.compile(r"^(?P<id>.+?)(?:_original_SAE)?\.pt$")

def filename_to_id(filename: str) -> str:
    m = ID_FROM_FILENAME.match(filename)
    if m: return m.group("id")
    return filename[:-3] if filename.endswith(".pt") else filename

def list_pt(d):
    return sorted(p for p in glob(os.path.join(d, "*.pt"))
                  if not os.path.basename(p).startswith("._"))

def safe_load_pt(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    return torch.load(path, map_location="cpu")

def human_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def fasta_ids(path):
    ids = []
    with open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids

def seqid_to_ac_key(seq_id: str) -> str:
    # POS: split on ':' → AC base; NEG: keep full if it contains '|'
    if '|' in seq_id:
        return seq_id
    return seq_id.split(':', 1)[0] if ':' in seq_id else seq_id

def build_id_to_fold_from_csv(csv_path: str):
    """Positive folds from GraphPart CSV (AC, cluster)."""
    df = pd.read_csv(csv_path)
    if "AC" not in df.columns or "cluster" not in df.columns:
        raise ValueError("CSV must contain 'AC' and 'cluster' columns.")
    cl = pd.to_numeric(df["cluster"], errors="coerce").round().astype("Int64")
    df = df.assign(cluster_int=cl)
    id2fold = {}
    for ac, c in zip(df["AC"].astype(str), df["cluster_int"]):
        if pd.isna(c):
            continue
        id2fold[ac] = int(c)
    return id2fold

def build_embed_index(embed_dir: str):
    """Return mapping: ID -> path (first occurrence)."""
    id_to_path = {}
    for p in list_pt(embed_dir):
        sid = filename_to_id(os.path.basename(p))
        id_to_path.setdefault(sid, p)
    return id_to_path

def compute_rankings_for_trainset(pos_files, neg_files, thresholds: torch.Tensor):
    """Seq-max pooling → F1 sweep → per-feature winning threshold & score."""
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    T = thresholds.numel()
    TP = FP = FN = None
    feat_dim = None

    def ensure_counters(v_feat):
        nonlocal TP, FP, FN, feat_dim
        if TP is None:
            feat_dim = v_feat.shape[0]
            TP = torch.zeros((T, feat_dim), dtype=torch.int64)
            FP = torch.zeros((T, feat_dim), dtype=torch.int64)
            FN = torch.zeros((T, feat_dim), dtype=torch.int64)

    def update_counts(v_feat, is_pos: bool):
        pred = (v_feat.unsqueeze(0) > thresholds.unsqueeze(1))  # (T,F)
        if is_pos:
            TP.add_(pred.to(torch.int64))
            FN.add_((~pred).to(torch.int64))
        else:
            FP.add_(pred.to(torch.int64))

    for flist, is_pos in ((pos_files, True), (neg_files, False)):
        for path in flist:
            try:
                x = safe_load_pt(path)
            except Exception:
                continue
            if x.ndim != 2:
                continue
            v = x.max(dim=0).values.to(torch.float32)  # (F,)
            ensure_counters(v)
            update_counts(v, is_pos)

    if TP is None:
        return pd.DataFrame(columns=["feature","winning_threshold","winning_F1_score"])

    rows = []
    for ti, thr in enumerate(thresholds.tolist()):
        tp = TP[ti].numpy()
        fp = FP[ti].numpy()
        fn = FN[ti].numpy()
        for feat in range(tp.shape[0]):
            TPi, FPi, FNi = int(tp[feat]), int(fp[feat]), int(fn[feat])
            prec = (TPi / (TPi + FPi)) if (TPi + FPi) else 0.0
            rec  = (TPi / (TPi + FNi)) if (TPi + FNi) else 0.0
            f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
            rows.append((feat, thr, f1))

    df = pd.DataFrame(rows, columns=["feature","threshold","F1_score"])
    best_idx = df.groupby("feature")["F1_score"].idxmax()
    best_df  = df.loc[best_idx, ["feature","threshold","F1_score"]].copy()
    best_df.rename(columns={"threshold":"winning_threshold",
                            "F1_score":"winning_F1_score"}, inplace=True)
    best_df.sort_values("feature", inplace=True, ignore_index=True)
    return best_df

# ── Main ───────────────────────────────────────────────────────────────
def main():
    # Load permitted IDs
    pos_ids = fasta_ids(POS_FASTA)
    neg_ids = fasta_ids(NEG_FASTA)
    print(f"[i] Permitted POS IDs: {len(pos_ids):,}")
    print(f"[i] Permitted NEG IDs: {len(neg_ids):,}")

    # Map positive IDs to folds via GraphPart CSV
    id2fold_pos = build_id_to_fold_from_csv(FOLDS_CSV)

    pos_pairs = []
    pos_missing_fold = 0
    for sid in pos_ids:
        ac = seqid_to_ac_key(sid)
        fold = id2fold_pos.get(ac, None)
        if fold is None or fold < 0 or fold > 9:
            pos_missing_fold += 1
            continue
        pos_pairs.append((sid, fold))
    if pos_missing_fold:
        print(f"[!] POS IDs without a valid fold assignment: {pos_missing_fold:,} (ignored)")

    # Create NEGATIVE 10 equal folds with deterministic seed
    random.seed(9427385)
    neg_ids_shuffled = neg_ids[:]  # copy
    random.shuffle(neg_ids_shuffled)

    N = len(neg_ids_shuffled)
    base = N // 10
    rem  = N % 10  # first 'rem' folds get one extra
    neg_assign = []
    idx = 0
    for fold in range(10):
        take = base + (1 if fold < rem else 0)
        for _ in range(take):
            neg_assign.append((neg_ids_shuffled[idx], fold))
            idx += 1
    assert idx == N

    # Save NEG fold map CSV
    os.makedirs(os.path.dirname(NEG_FOLDS_OUT), exist_ok=True)
    with open(NEG_FOLDS_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry", "fold"])
        for sid, fold in neg_assign:
            w.writerow([sid, fold])
    print(f"[✓] Wrote negative 10-fold map → {NEG_FOLDS_OUT}")
    print("    Fold sizes: " + ", ".join(
        f"{fold}:{sum(1 for _,fd in neg_assign if fd==fold):,}" for fold in range(10)
    ))

    # Build embedding indices (ID -> path)
    pos_id2path = build_embed_index(POS_EMB_DIR)
    neg_id2path = build_embed_index(NEG_EMB_DIR)

    # Organize by fold
    byfold_pos = {k: [] for k in range(10)}
    for sid, f in pos_pairs:
        byfold_pos[f].append(sid)

    byfold_neg = {k: [] for k in range(10)}
    for sid, f in neg_assign:
        byfold_neg[f].append(sid)

    # ── Overall missing-embedding audit (once) ─────────────────────────
    pos_ids_set = {sid for sid, _ in pos_pairs}
    neg_ids_set = {sid for sid, _ in neg_assign}

    overall_pos_missing = sorted(pos_ids_set - set(pos_id2path.keys()))
    overall_neg_missing = sorted(neg_ids_set - set(neg_id2path.keys()))

    if overall_pos_missing:
        path = os.path.join(OUT_DIR, "overall_POS_missing_embedding_ids.txt")
        with open(path, "w") as f:
            f.write("\n".join(overall_pos_missing))
        print(f"[overall] POS permitted(train+test): {len(pos_ids_set):,} | missing embeds: {len(overall_pos_missing):,} → {path}")
    else:
        print(f"[overall] POS permitted(train+test): {len(pos_ids_set):,} | missing embeds: 0")

    if overall_neg_missing:
        path = os.path.join(OUT_DIR, "overall_NEG_missing_embedding_ids.txt")
        with open(path, "w") as f:
            f.write("\n".join(overall_neg_missing))
        print(f"[overall] NEG permitted(train+test): {len(neg_ids_set):,} | missing embeds: {len(overall_neg_missing):,} → {path}")
    else:
        print(f"[overall] NEG permitted(train+test): {len(neg_ids_set):,} | missing embeds: 0")

    # Loop over excluded fold k
    for k in range(10):
        print(f"\n=== Excluding fold {k} (training on folds != {k}) ===")
        # Collect training IDs
        train_pos_ids = [sid for f, lst in byfold_pos.items() if f != k for sid in lst]
        train_neg_ids = [sid for f, lst in byfold_neg.items() if f != k for sid in lst]

        # Per-fold missing-embedding audit
        fold_pos_missing = sorted(set(train_pos_ids) - set(pos_id2path.keys()))
        fold_neg_missing = sorted(set(train_neg_ids) - set(neg_id2path.keys()))

        if fold_pos_missing:
            path_pos = os.path.join(OUT_DIR, f"fold_{k}_POS_missing_embedding_ids.txt")
            with open(path_pos, "w") as f:
                f.write("\n".join(fold_pos_missing))
            print(f"[fold {k}] POS train IDs: {len(train_pos_ids):,} | missing embeds: {len(fold_pos_missing):,} → {path_pos}")
        else:
            print(f"[fold {k}] POS train IDs: {len(train_pos_ids):,} | missing embeds: 0")

        if fold_neg_missing:
            path_neg = os.path.join(OUT_DIR, f"fold_{k}_NEG_missing_embedding_ids.txt")
            with open(path_neg, "w") as f:
                f.write("\n".join(fold_neg_missing))
            print(f"[fold {k}] NEG train IDs: {len(train_neg_ids):,} | missing embeds: {len(fold_neg_missing):,} → {path_neg}")
        else:
            print(f"[fold {k}] NEG train IDs: {len(train_neg_ids):,} | missing embeds: 0")

        # Map to existing embedding file paths (skip missing)
        train_pos_files = [pos_id2path[sid] for sid in train_pos_ids if sid in pos_id2path]
        train_neg_files = [neg_id2path[sid] for sid in train_neg_ids if sid in neg_id2path]
        print(f"[fold {k}] using POS embeddings: {len(train_pos_files):,} | NEG embeddings: {len(train_neg_files):,}")

        if not train_pos_files or not train_neg_files:
            print(f"[fold {k}] WARNING: empty POS or NEG training set after filtering—writing empty ranking.")
            empty = pd.DataFrame(columns=["feature","winning_threshold","winning_F1_score"])
            out_csv = os.path.join(OUT_DIR, f"feature_rank_excluded_fold_{k}.csv")
            empty.to_csv(out_csv, index=False)
            # Also emit an empty ranking file for consistency
            out_rank = os.path.join(OUT_DIR, f"feature_ranking_excluded_fold_{k}.csv")
            pd.DataFrame(columns=["rank","feature","F1_score"]).to_csv(out_rank, index=False)
            print(f"[fold {k}] wrote EMPTY ranking files → {out_csv} , {out_rank}")
            continue

        # Compute rankings on training set
        t0 = time.time()
        best_df = compute_rankings_for_trainset(train_pos_files, train_neg_files, THRESHOLDS)
        dt = time.time() - t0

        # Save the best-per-feature table
        out_csv = os.path.join(OUT_DIR, f"winning_thresholds_excluded_fold_{k}.csv")
        best_df.to_csv(out_csv, index=False)

        # Build and save the ACTUAL RANKING (sorted by F1 desc, 1-based rank)
        ranking_df = (
            best_df[["feature", "winning_F1_score"]]
            .rename(columns={"winning_F1_score": "F1_score"})
            .sort_values("F1_score", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))
        out_rank = os.path.join(OUT_DIR, f"feature_ranking_excluded_fold_{k}.csv")
        ranking_df.to_csv(out_rank, index=False)

        print(f"[fold {k}] features ranked: {len(best_df):,} | time {human_time(dt)}")
        print(f"[fold {k}] wrote → {out_csv}")
        print(f"[fold {k}] wrote ranking → {out_rank}")

if __name__ == "__main__":
    main()
