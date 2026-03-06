#!/usr/bin/env python3
import os, sys, glob
import torch, joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression  # keep importable

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 900          # ← your run-6 model uses 900 features
PRED_THRESH      = 0.5
run              = 6

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis"

# Known-peptide embeddings + FASTA
EMB_DIR    = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/known_pep_finetuned_SAE_embeddings/run_{run}/"
FASTA_PATH = "/Volumes/T7 Shield/uniprot_known_neuropep_sequences.fasta"

# (Optional) positives for accuracy; leave as-is if you don’t have this file
POS_CSV = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_positive_known_peps.csv"

# Model + features (trained in your final-model script for run_6 / 900)
models_root       = os.path.join(
    DATA_ROOT, "classifier_results", f"run_{run}", "logistic_regression_models",
    f"graphpart_{percent_identity}"
)
model_path        = os.path.join(models_root, f"final_logreg_model_{num_features}_features_balanced.joblib")
features_csv_path = os.path.join(models_root, f"final_logreg_features_{num_features}.csv")

# Output
out_dir = os.path.join(DATA_ROOT, "classifier_results", "logreg_known_peptides", f"graphpart_{percent_identity}")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f"known_peptides_logreg_probs_predictions_{num_features}.csv")

# ─── Helpers ───────────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict:
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    base = os.path.basename(path)
    if "|" in base:
        parts = base.split("|")
        return parts[1] if len(parts) >= 3 else None
    return os.path.splitext(base)[0]

# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # 0) Optional ground-truth map
    if not os.path.isfile(POS_CSV):
        print(f"⚠️  Positives CSV not found: {POS_CSV} — proceeding without accuracy.")
        pos_sites_map = {}
    else:
        pos_df = pd.read_csv(POS_CSV)
        if "residue_number" in pos_df.columns:
            pos_df["residue_number"] = pos_df["residue_number"].astype(int)
            pos_sites_map = (
                pos_df.groupby("Entry")["residue_number"]
                .apply(lambda s: set(s.tolist()))
                .to_dict()
            )
        else:
            pos_sites_map = {}

    # 1) Load model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Logistic regression model not found: {model_path}")
    clf = joblib.load(model_path)
    print(f"Model: {model_path}")
    print(f"Features CSV: {features_csv_path}")
    print(f"Run used for embeddings: {EMB_DIR}")
    # Debug info
    try:
        print("classes_:", clf.classes_)
        if hasattr(clf, "coef_"):
            print("coef_.shape:", clf.coef_.shape)
    except Exception:
        pass

    # 2) Feature indices used
    if not os.path.isfile(features_csv_path):
        raise FileNotFoundError(f"Features CSV not found: {features_csv_path}")
    feats_df  = pd.read_csv(features_csv_path)
    top_feats = feats_df["feature"].astype(int).tolist()
    if len(top_feats) != num_features:
        print(f"⚠️ Feature list has {len(top_feats)} indices but num_features={num_features}. Proceeding anyway.")
    print(f"✔ Using {len(top_feats)} features (max index = {max(top_feats)})")

    # 3) Load FASTA sequences → {Entry: seq}
    if not os.path.isfile(FASTA_PATH):
        raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # 4) Score each known peptide embedding
    rows = []
    total_correct = 0
    total_count   = 0

    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found in {EMB_DIR}")

    for emb_fp in emb_paths:
        entry = entry_from_embedding_path(emb_fp)
        if not entry:
            print(f"⚠️  Could not parse Entry from filename: {emb_fp}; skipping.")
            continue

        arr = torch.load(emb_fp, map_location="cpu").numpy()  # (L, D)
        L, D = arr.shape
        if max(top_feats) >= D:
            print(f"⚠️  Skipping {entry}: feature index {max(top_feats)} exceeds embedding dim {D}")
            continue

        X_seq = arr[:, top_feats]  # (L, num_features)

        # debug: check variance and decision scores to detect saturation
        print(f"{entry}: X_seq shape {X_seq.shape}, var[0:5]={np.var(X_seq, axis=0)[:5]}")
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_seq)
            print(f"{entry}: decision scores min/med/max = "
                  f"{float(np.min(scores)):.3f}/{float(np.median(scores)):.3f}/{float(np.max(scores)):.3f}")
        # compute probabilities
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_seq)[:, 1]
        else:
            scores = clf.decision_function(X_seq)
            probs  = 1 / (1 + np.exp(-scores))
        print(f"{entry}: probs min/med/max = "
              f"{float(np.min(probs)):.3f}/{float(np.median(probs)):.3f}/{float(np.max(probs)):.3f}")

        preds = (probs >= PRED_THRESH).astype(int)

        seq = entry_to_seq.get(entry, None)
        if seq is None:
            print(f"⚠️  No FASTA sequence for {entry}; residues will be 'X'.")
            seq = "X" * L

        if len(seq) != L:
            print(f"⚠️  Length mismatch for {entry}: FASTA={len(seq)} EMB={L}. Using min length.")
        use_L = min(L, len(seq))

        pos_set = pos_sites_map.get(entry, set())

        for i in range(use_L):
            rows.append({
                "Entry": entry,
                "position": i + 1,
                "residue": seq[i],
                "probability": float(probs[i]),
                "predictions": int(preds[i]),
            })
            if pos_sites_map:
                y_true = 1 if (i + 1) in pos_set else 0
                total_correct += int(y_true == preds[i])
                total_count   += 1

    # 5) Save CSV
    out_df = pd.DataFrame(rows, columns=["Entry", "position", "residue", "probability", "predictions"])
    out_df.to_csv(out_csv, index=False)
    print(f"✔ Saved probabilities + predictions for {len(out_df)} positions → {out_csv}")

    if pos_sites_map and total_count > 0:
        acc = total_correct / total_count * 100.0
        print(f"✔ Overall accuracy @ threshold={PRED_THRESH:.2f}: {acc:.2f}% "
              f"({total_correct}/{total_count} correct)")

if __name__ == "__main__":
    main()
