#!/usr/bin/env python3
# F1 for max-pooled ESM-2 embeddings, restricted to permitted IDs

import os, time
from glob import glob
import torch
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
positive_dir = "/Volumes/T7 Shield/ESM2_embeddings_training/positive"
negative_dir = "/Volumes/T7 Shield/ESM2_embeddings_training/negative"
output_dir   = "/Volumes/T7 Shield/ESM2_F1_scores"
os.makedirs(output_dir, exist_ok=True)

pos_fasta_allowed = "../../data/combined_l18_positive_final.fasta"
neg_fasta_allowed = "../../data/negative_l18_dataset_final.fasta"

# ── Thresholds ─────────────────────────────────────────────────────────
thresholds = torch.tensor([0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=torch.float32)
T = thresholds.numel()

print("\n========== ESM-2 F1 ANALYSIS ==========")
print("Step 1: Load permitted FASTA IDs")
print("Step 2: Match ESM-2 embeddings to permitted IDs")
print("Step 3: Accumulate TP / FP / FN counts")
print("Step 4: Compute feature-wise F1 scores\n")

# ── Helpers ────────────────────────────────────────────────────────────
def safe_div(a, b): 
    return (a / b) if b else 0.0

def list_pt(d):
    return sorted(
        p for p in glob(os.path.join(d, "*.pt"))
        if not os.path.basename(p).startswith("._")
    )

def safe_load(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    return torch.load(path, map_location="cpu")

def human_time(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

torch.set_grad_enabled(False)
torch.set_num_threads(1)

# ── FASTA allowlists ───────────────────────────────────────────────────
def fasta_ids(path):
    ids = set()
    with open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids

allowed_pos_ids = fasta_ids(pos_fasta_allowed)
allowed_neg_ids = fasta_ids(neg_fasta_allowed)

print(f"[allowlists] positives: {len(allowed_pos_ids):,}")
print(f"[allowlists] negatives: {len(allowed_neg_ids):,}\n")

# ── Metrics accumulators ───────────────────────────────────────────────
TP = FP = FN = None
feat_dim = None

def ensure_counters(v_feat):
    global TP, FP, FN, feat_dim
    if TP is None:
        feat_dim = v_feat.shape[0]
        TP = torch.zeros((T, feat_dim), dtype=torch.int64)
        FP = torch.zeros((T, feat_dim), dtype=torch.int64)
        FN = torch.zeros((T, feat_dim), dtype=torch.int64)
        print(f"[init] feature_dim={feat_dim}  counters={list(TP.shape)}")

def update_counts(v_feat, is_positive):
    pred = (v_feat.unsqueeze(0) > thresholds.unsqueeze(1))  # (T, F)
    if is_positive:
        TP.add_(pred.to(torch.int64))
        FN.add_((~pred).to(torch.int64))
    else:
        FP.add_(pred.to(torch.int64))

# ── Match embeddings to FASTA IDs ──────────────────────────────────────
def select_files_and_check_missing(embed_dir, allowed_ids, tag):
    print(f"[{tag}] Scanning embedding directory: {embed_dir}")
    all_files = list_pt(embed_dir)
    print(f"[{tag}] .pt files on disk: {len(all_files):,}")

    id_to_path = {}
    for p in all_files:
        try:
            obj = safe_load(p)
        except Exception:
            continue

        if not isinstance(obj, dict) or "entry" not in obj:
            continue

        true_id = obj["entry"]
        id_to_path[true_id] = p

    print(f"[{tag}] Unique embedding IDs loaded: {len(id_to_path):,}")
    if id_to_path:
        print(f"[{tag}] Example embedding ID: {next(iter(id_to_path))}")

    missing = allowed_ids - set(id_to_path)
    keep_files = [id_to_path[i] for i in sorted(allowed_ids & set(id_to_path))]

    miss_path = os.path.join(output_dir, f"{tag}_missing_ids.txt")
    with open(miss_path, "w") as f:
        for i in sorted(missing):
            f.write(i + "\n")

    print(
        f"[{tag}] permitted: {len(allowed_ids):,} | "
        f"embeddings found: {len(keep_files):,} | "
        f"MISSING: {len(missing):,} → {miss_path}\n"
    )

    return keep_files, len(missing)

# ── Process embeddings ────────────────────────────────────────────────
def process_files(file_list, is_positive, tag, log_every=1000):
    print(f"\n[{tag}] Starting processing of {len(file_list):,} embeddings")

    skipped_load = skipped_shape = 0
    start = time.time()
    n = len(file_list)

    for i, path in enumerate(file_list, 1):
        try:
            obj = safe_load(path)
        except Exception:
            skipped_load += 1
            continue

        if not isinstance(obj, dict) or "embedding" not in obj:
            skipped_shape += 1
            continue

        v = obj["embedding"]
        if v.ndim != 1:
            skipped_shape += 1
            continue

        # First-embedding sanity check
        if i == 1:
            print(f"[{tag}] First embedding entry: {obj['entry']}")
            print(f"[{tag}] First embedding shape: {v.shape}")

        v = v.to(torch.float32)
        ensure_counters(v)
        update_counts(v, is_positive=is_positive)

        if (i % log_every) == 0 or i == n:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else float("inf")
            eta = (n - i) / rate if rate > 0 else 0
            print(
                f"  [{tag}] {i:>6}/{n:<6} | "
                f"{rate:6.1f} seq/s | "
                f"elapsed {human_time(elapsed)} | ETA {human_time(eta)}"
            )

    used = n - skipped_load - skipped_shape
    dur = time.time() - start
    print(
        f"[{tag}] usable={used:,}  skipped_load={skipped_load:,}  "
        f"skipped_shape={skipped_shape:,}  time={human_time(dur)}\n"
    )
    return dict(usable=used, candidates=n)

# ── Run ─────────────────────────────────────────────────────────────────
print(
    "== F1 (ESM-2 max-pooled) with permitted-ID filtering ==\nthresholds:",
    thresholds.tolist(), "\n"
)

pos_files, pos_missing = select_files_and_check_missing(positive_dir, allowed_pos_ids, "POS")
neg_files, neg_missing = select_files_and_check_missing(negative_dir, allowed_neg_ids, "NEG")

pos_stats = process_files(pos_files, is_positive=True,  tag="POS")
neg_stats = process_files(neg_files, is_positive=False, tag="NEG")

print("\n[INFO] Finished accumulating counts")
print(f"[INFO] TP sum: {TP.sum().item()}")
print(f"[INFO] FP sum: {FP.sum().item()}")
print(f"[INFO] FN sum: {FN.sum().item()}")

# ── Compute metrics ────────────────────────────────────────────────────
rows = []
for ti, thr in enumerate(thresholds.tolist()):
    tp, fp, fn = TP[ti].numpy(), FP[ti].numpy(), FN[ti].numpy()
    for feat in range(tp.shape[0]):
        TPi, FPi, FNi = int(tp[feat]), int(fp[feat]), int(fn[feat])
        prec = safe_div(TPi, TPi + FPi)
        rec  = safe_div(TPi, TPi + FNi)
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({
            "feature": feat,
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "F1_score": f1,
            "TP": TPi,
            "FP": FPi,
            "FN": FNi,
        })

df = pd.DataFrame(rows).sort_values(["threshold", "feature"])
out_csv = os.path.join(output_dir, "ESM2_F1_seqmax_all_thresholds_filtered.csv")
df.to_csv(out_csv, index=False)

print("\n== Summary ================================")
print(f"features:                    {TP.shape[1]}")
print(f"thresholds:                  {thresholds.tolist()}")
print(f"POS missing permitted IDs:   {pos_missing:,}")
print(f"NEG missing permitted IDs:   {neg_missing:,}")
print(f"POS usable/candidates:       {pos_stats['usable']:,}/{pos_stats['candidates']:,}")
print(f"NEG usable/candidates:       {neg_stats['usable']:,}/{neg_stats['candidates']:,}")
print(f"output CSV:                  {out_csv}")
print("===========================================")
