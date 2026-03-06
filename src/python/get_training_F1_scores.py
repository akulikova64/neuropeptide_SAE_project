#!/usr/bin/env python3
# F1 with per-sequence max pooling, restricted to permitted IDs,
# and ONLY checking that all permitted IDs have embeddings.

import os, time, re
from glob import glob
import torch
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
positive_dir = "/Volumes/T7 Shield/layer_18_embeddings/"
negative_dir = "/Volumes/T7 Shield/layer_18_negative_embeddings/"
output_dir   = "/Volumes/T7 Shield/layer_18_F1_scores/"
os.makedirs(output_dir, exist_ok=True)

pos_fasta_allowed = "../../data/combined_l18_positive_final.fasta"
neg_fasta_allowed = "../../data/negative_l18_dataset_final.fasta"

# ── Thresholds ─────────────────────────────────────────────────────────
thresholds = torch.tensor([0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=torch.float32)
T = thresholds.numel()

# ── Helpers ────────────────────────────────────────────────────────────
def safe_div(a, b): return (a / b) if b else 0.0

def list_pt(d):
    return sorted(p for p in glob(os.path.join(d, "*.pt"))
                  if not os.path.basename(p).startswith("._"))

def safe_load(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    return torch.load(path, map_location="cpu")

def human_time(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

torch.set_grad_enabled(False)
torch.set_num_threads(1)

# ID rules:
#  • FASTA: first token after '>' (keep ':' and '|')
#  • Filename: strip "_original_SAE.pt" or just ".pt"
ID_FROM_FILENAME = re.compile(r"^(?P<id>.+?)(?:_original_SAE)?\.pt$")

def filename_to_id(filename):
    m = ID_FROM_FILENAME.match(filename)
    if m: return m.group("id")
    return filename[:-3] if filename.endswith(".pt") else filename

def fasta_ids(path):
    ids = set()
    with open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids

allowed_pos_ids = fasta_ids(pos_fasta_allowed)
allowed_neg_ids = fasta_ids(neg_fasta_allowed)
print(f"[allowlists] positives: {len(allowed_pos_ids):,} | negatives: {len(allowed_neg_ids):,}\n")

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
    pred = (v_feat.unsqueeze(0) > thresholds.unsqueeze(1))  # (T,F)
    if is_positive:
        TP.add_(pred.to(torch.int64))
        FN.add_((~pred).to(torch.int64))
    else:
        FP.add_(pred.to(torch.int64))

def select_files_and_check_missing(embed_dir, allowed_ids, tag):
    """Return list of files to use (intersection) and write only the missing-ID list."""
    all_files = list_pt(embed_dir)
    have_ids = set()
    id_to_path = {}
    for p in all_files:
        sid = filename_to_id(os.path.basename(p))
        have_ids.add(sid)
        id_to_path.setdefault(sid, p)

    missing = allowed_ids - have_ids
    keep_ids = allowed_ids & have_ids
    keep_files = [id_to_path[i] for i in sorted(keep_ids)]

    miss_path = os.path.join(output_dir, f"{tag}_missing_ids.txt")
    with open(miss_path, "w") as f:
        for i in sorted(missing): f.write(i + "\n")

    print(f"[{tag}] permitted: {len(allowed_ids):,} | embeddings found: {len(keep_ids):,} "
          f"| MISSING: {len(missing):,} → {miss_path}\n")

    return keep_files, len(missing)

def process_files(file_list, is_positive, tag, log_every=1000):
    skipped_load = skipped_shape = 0
    start = time.time()
    n = len(file_list)
    for i, path in enumerate(file_list, 1):
        try:
            x = safe_load(path)  # expect (L,F)
        except Exception as e:
            skipped_load += 1
            if skipped_load <= 5:
                print(f"  ⚠️  skip load {os.path.basename(path)} ({e})")
            continue
        if x.ndim != 2:
            skipped_shape += 1
            if skipped_shape <= 5:
                print(f"  ⚠️  skip shape {os.path.basename(path)} (shape {tuple(x.shape)})")
            continue

        v = x.max(dim=0).values.to(torch.float32)
        ensure_counters(v)
        update_counts(v, is_positive=is_positive)

        if (i % log_every) == 0 or i == n:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else float("inf")
            eta = (n - i)/rate if rate>0 else 0
            print(f"  [{tag}] {i:>6}/{n:<6} | {rate:6.1f} seq/s | elapsed {human_time(elapsed)} | ETA {human_time(eta)}")

    used = n - skipped_load - skipped_shape
    dur = time.time() - start
    rate = used / dur if dur > 0 else 0.0
    print(f"[{tag}] usable={used:,}  skipped_load={skipped_load:,}  skipped_shape={skipped_shape:,}  time={human_time(dur)}  rate={rate:,.1f} seq/s\n")
    return dict(usable=used, candidates=n, skipped_load=skipped_load, skipped_shape=skipped_shape, time=dur, rate=rate)

# ── Run ─────────────────────────────────────────────────────────────────
print("== F1 (seq-max) per feature with permitted-ID filtering & missing-check ==\nthresholds:",
      thresholds.tolist(), "\n")

pos_files, pos_missing = select_files_and_check_missing(positive_dir, allowed_pos_ids, tag="POS")
neg_files, neg_missing = select_files_and_check_missing(negative_dir, allowed_neg_ids, tag="NEG")

if pos_missing or neg_missing:
    print("⚠️  Some permitted IDs are missing embeddings. See *_missing_ids.txt files above.")
# You can uncomment the next two lines to hard-stop if anything is missing:
# if pos_missing or neg_missing:
#     raise SystemExit("Permitted IDs missing embeddings—fix before proceeding.")

if not pos_files:
    raise RuntimeError("No positive embeddings match the permitted positive IDs.")
if not neg_files:
    raise RuntimeError("No negative embeddings match the permitted negative IDs.")

pos_stats = process_files(pos_files, is_positive=True,  tag="POS")
neg_stats = process_files(neg_files, is_positive=False, tag="NEG")

if TP is None:
    raise RuntimeError("No valid tensors processed—check ID matching and embedding file integrity.")

# ── Compute metrics ────────────────────────────────────────────────────
rows = []
for ti, thr in enumerate(thresholds.tolist()):
    tp, fp, fn = TP[ti].numpy(), FP[ti].numpy(), FN[ti].numpy()
    for feat in range(tp.shape[0]):
        TPi, FPi, FNi = int(tp[feat]), int(fp[feat]), int(fn[feat])
        prec = safe_div(TPi, TPi + FPi)
        rec  = safe_div(TPi, TPi + FNi)
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        rows.append({"feature": feat, "threshold": thr,
                     "precision": prec, "recall": rec, "F1_score": f1,
                     "TP": TPi, "FP": FPi, "FN": FNi})

df = pd.DataFrame(rows).sort_values(["threshold","feature"])
out_csv = os.path.join(output_dir, "F1_seqmax_all_thresholds_filtered.csv")
df.to_csv(out_csv, index=False)

print("== Summary ================================")
print(f"features:                    {TP.shape[1]}")
print(f"thresholds:                  {thresholds.tolist()}")
print(f"POS missing permitted IDs:   {pos_missing:,}")
print(f"NEG missing permitted IDs:   {neg_missing:,}")
print(f"POS usable/candidates:       {pos_stats['usable']:,}/{pos_stats['candidates']:,} (skipped_load={pos_stats['skipped_load']:,}, skipped_shape={pos_stats['skipped_shape']:,})")
print(f"NEG usable/candidates:       {neg_stats['usable']:,}/{neg_stats['candidates']:,} (skipped_load={neg_stats['skipped_load']:,}, skipped_shape={neg_stats['skipped_shape']:,})")
print(f"output CSV:                  {out_csv}")
print("===========================================")
