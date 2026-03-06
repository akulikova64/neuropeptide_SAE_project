#!/usr/bin/env python3
import os
import re
import gc
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn.metrics import accuracy_score  # kept on purpose
from scipy.signal import find_peaks

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40
max_iter         = 20
feature_step     = 10
feature_max      = 255

# Peak-detection settings (tune on validation)
PEAK_MIN_DISTANCE = 1      # minimum distance (in residues) between peaks
PEAK_MIN_PROM     = 0.10   # required prominence of a peak
PEAK_MIN_WIDTH    = 1      # minimum peak width (in samples)
PEAK_MIN_HEIGHT   = 0.05   # minimum peak height
PEAK_NEIGHBOR_RATIO = 0.5  # neighbor must be >= this fraction of peak height

# ─── Peak helpers ──────────────────────────────────────────────────────────
def detect_sites_by_peaks(
    probs: np.ndarray,
    min_distance: int = 1,
    min_prom: float  = 0.05,
    min_width: int   = 1,
    min_height: float | None = None,
    neighbor_ratio: float = 0.5,  # neighbor must be >= neighbor_ratio * peak_height
):
    """
    Find peaks, then also mark immediate neighbors whose probability is at least
    `neighbor_ratio` times the peak's probability.
    Returns (y_pred, peaks, props) where y_pred is a 0/1 vector.
    """
    kwargs = dict(distance=min_distance, prominence=min_prom, width=min_width)
    if min_height is not None:
        kwargs["height"] = min_height

    peaks, props = find_peaks(probs, **kwargs)
    L = len(probs)
    y_pred = np.zeros(L, dtype=int)

    # mark peaks
    if peaks.size:
        y_pred[peaks] = 1

    # mark neighbors meeting relative threshold
    for p in peaks:
        peak_h = float(probs[p])
        thr = neighbor_ratio * peak_h
        for q in (p - 1, p + 1):
            if 0 <= q < L and probs[q] >= thr:
                y_pred[q] = 1

    return y_pred, peaks, props

def metrics_from_counts(TP, FP, TN, FN):
    total = TP + FP + TN + FN
    overall = (TP + TN) / total if total else np.nan
    pos_acc = TP / (TP + FN)     if (TP + FN) else np.nan  # within-class accuracy of positives
    neg_acc = TN / (TN + FP)     if (TN + FP) else np.nan  # within-class accuracy of negatives
    return overall, pos_acc, neg_acc

def evaluate_sequence_by_peaks(
    probs: np.ndarray,
    y_true_seq: np.ndarray,
    min_distance: int = PEAK_MIN_DISTANCE,
    min_prom: float  = PEAK_MIN_PROM,
    min_width: int   = PEAK_MIN_WIDTH,
    min_height: float | None = PEAK_MIN_HEIGHT,
    neighbor_ratio: float = PEAK_NEIGHBOR_RATIO,
):
    """
    probs: np.array of length L with P(y=1) per residue
    y_true_seq: np.array of 0/1 ground-truth per position (length L)
    Returns TP, FP, TN, FN for this sequence based on peak calls.
    """
    y_true = y_true_seq.astype(int)
    y_pred, _, _ = detect_sites_by_peaks(
        probs,
        min_distance=min_distance,
        min_prom=min_prom,
        min_width=min_width,
        min_height=min_height,
        neighbor_ratio=neighbor_ratio,
    )
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN

# ─── Static roots ──────────────────────────────────────────────────────────
DATA_ROOT    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_ROOT     = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
POS_CSV      = os.path.join(DATA_ROOT, "input_data", "group_2_positive_toxin_neuro.csv")
FOLDS_ROOT   = os.path.join(DATA_ROOT, "input_data", f"folds_{percent_identity}")

for test_fold in test_folds:
    print(f"\n=== EXCLUDING FOLD {test_fold} AS TEST SET ===")

    # ─── Paths for this fold ────────────────────────────────────────────────
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/"
        f"excluded_fold_{test_fold}/"
        f"feature_ranking_excluded_fold_{test_fold}.csv"
    )
    models_root = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"classifier_results/conditional_random_field_models_3/"
        f"graphpart_{percent_identity}/excluded_fold_{test_fold}"
    )
    csv_root = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"classifier_results/crf_csv_results_3/"
        f"graphpart_{percent_identity}/excluded_fold_{test_fold}"
    )
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(csv_root,    exist_ok=True)

    # ─── Load feature ranking ────────────────────────────────────────────────
    feat_df       = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_list  = feat_df["feature"].tolist()

    # ─── Load positive sites ─────────────────────────────────────────────────
    pos_df        = pd.read_csv(POS_CSV)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)

    # ─── Determine allowed entries (exclude test_fold) ───────────────────────
    test_fp = os.path.join(FOLDS_ROOT, f"fold_{test_fold}.txt")
    with open(test_fp, encoding="utf-8", errors="ignore") as fh:
        test_entries = set(fh.read().split())

    # Collect all entries that ever appear in your positives CSV
    pos_entries = set(pos_df["Entry"])

    # Build seq_dict for entries that are both in pos_entries AND not in the test fold
    seq_dict = {}
    for emb_fn in os.listdir(EMB_ROOT):
        entry = emb_fn.replace("_original_SAE.pt", "")
        if entry in test_entries or entry not in pos_entries:
            continue
        full_path = os.path.join(EMB_ROOT, emb_fn)
        arr       = torch.load(full_path, map_location="cpu").numpy()
        L         = arr.shape[0]
        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"])
        labels    = [str(1 if (i+1) in pos_sites else 0) for i in range(L)]
        seq_dict[entry] = {"raw": arr, "y": labels}

    entries = list(seq_dict.keys())
    print(f"  {len(entries)} entries after excluding fold {test_fold}")

    # ─── Build full_dict_seqs once ───────────────────────────────────────────
    full_dict_seqs = []
    for entry in entries:
        arr = seq_dict[entry]["raw"]
        seq_feats = [
            {f"f{i}": float(arr[pos,i]) for i in feature_list}
            for pos in range(arr.shape[0])
        ]
        full_dict_seqs.append(seq_feats)

    Y_raw = [seq_dict[e]["y"] for e in entries]

    # ─── Gather inner-folds 0–9 minus test_fold ─────────────────────────────
    all_folds = list(range(10))
    non_ex_folds = [f for f in all_folds if f != test_fold]
    nums = "|".join(str(f) for f in non_ex_folds)
    pattern = re.compile(rf"^fold_(?:{nums})\.txt$")
    fold_files = sorted(
        fn for fn in os.listdir(FOLDS_ROOT)
        if pattern.match(fn)
    )
    print(f"  Inner CV folds: {fold_files}")

    # ─── Nested CV: vary num_features ───────────────────────────────────────
    records = []
    for num_feat in range(5, feature_max+1, feature_step):
        print(f"\n • num_features = {num_feat}")
        keys    = {f"f{i}" for i in feature_list[:num_feat]}
        X_seqs  = [
            [{k: token[k] for k in keys} for token in seq]
            for seq in full_dict_seqs
        ]

        accs, pos_accs, neg_accs = [], [], []
        for fold_file in fold_files:
            # validation entries for this inner fold
            path = os.path.join(FOLDS_ROOT, fold_file)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                val_entries = set(fh.read().split())
            # build train/test indices
            val_idx = [i for i,e in enumerate(entries) if e in val_entries]
            trn_idx = [i for i in range(len(entries)) if i not in val_idx]

            X_tr = [X_seqs[i] for i in trn_idx]
            y_tr = [Y_raw[i]   for i in trn_idx]
            X_te = [X_seqs[i] for i in val_idx]
            y_te = [Y_raw[i]   for i in val_idx]

            crf = CRF(
                algorithm="lbfgs",
                max_iterations=max_iter,
                all_possible_transitions=True
            )
            crf.fit(X_tr, y_tr)

            # ── Peak-based evaluation using CRF marginals ─────────────────
            TP = FP = TN = FN = 0
            for i_seq in range(len(X_te)):
                mlist = crf.predict_marginals_single(X_te[i_seq])
                probs = np.array([m.get("1", 0.0) for m in mlist], dtype=float)
                y_true_seq = np.array([int(v) for v in y_te[i_seq]], dtype=int)

                tp, fp, tn, fn = evaluate_sequence_by_peaks(
                    probs,
                    y_true_seq,
                    min_distance=PEAK_MIN_DISTANCE,
                    min_prom=PEAK_MIN_PROM,
                    min_width=PEAK_MIN_WIDTH,
                    min_height=PEAK_MIN_HEIGHT,
                    neighbor_ratio=PEAK_NEIGHBOR_RATIO,
                )
                TP += tp; FP += fp; TN += tn; FN += fn

            overall, pos_acc, neg_acc = metrics_from_counts(TP, FP, TN, FN)
            accs.append(overall); pos_accs.append(pos_acc); neg_accs.append(neg_acc)

            # cleanup this fold
            del crf, X_tr, y_tr, X_te, y_te
            gc.collect()

        avg_acc = float(np.nanmean(accs)) * 100.0
        avg_pos = float(np.nanmean(pos_accs)) * 100.0
        avg_neg = float(np.nanmean(neg_accs)) * 100.0

        print(f"   ↳ avg inner-fold acc = {avg_acc:.2f}% | pos_acc = {avg_pos:.2f}% | neg_acc = {avg_neg:.2f}%")

        records.append({
            "num_features": num_feat,
            "val_acc":      f"{avg_acc:.2f}",
            "pos_acc":      f"{avg_pos:.2f}",
            "neg_acc":      f"{avg_neg:.2f}"
        })

        # ─── Train final CRF on all non-test entries ────────────────────────
        crf_final = CRF(
            algorithm="lbfgs",
            max_iterations=max_iter,
            all_possible_transitions=True
        )
        crf_final.fit(X_seqs, Y_raw)
        joblib.dump(
            crf_final,
            os.path.join(models_root, f"crf_excl_fold{test_fold}_{num_feat}.joblib")
        )
        del crf_final
        gc.collect()

    # ─── Save nested‐CV results CSV ─────────────────────────────────────────
    out_csv = os.path.join(csv_root, f"crf_results_excluded_fold_{test_fold}.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"✔ Saved CRF CV results → {out_csv}")
