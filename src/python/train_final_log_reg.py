#!/usr/bin/env python3
import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 900
max_iter         = 1000
run              = 6
USE_CALIBRATION  = True  # set False if you want to skip probability calibration
CALIB_CV         = 5

# ─── Paths ───────────────────────────────────────────────────────────────
DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis"
EMB_ROOT   = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_finetuned_SAE_embeddings/run_{run}/"
POS_CSV    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_positive_toxin_neuro.csv"
NEG_CSV    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_negative_toxin_neuro.csv"
FOLDS_ROOT = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_{percent_identity}"

ranking_csv  = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/F1_scores_hybrid/F1_scores_graphpart_40/feature_ranking_all_folds_run_{run}.csv"

models_root      = os.path.join(
    DATA_ROOT, "classifier_results", f"run_{run}", "logistic_regression_models",
    f"graphpart_{percent_identity}"
)
os.makedirs(models_root, exist_ok=True)
model_out        = os.path.join(models_root, f"final_logreg_model_{num_features}_features_balanced.joblib")
features_out_csv = os.path.join(models_root, f"final_logreg_features_{num_features}.csv")

def find_embedding_path(entry: str):
    for name in (f"{entry}.pt", f"{entry}_original_SAE.pt", f"{entry}_finetuned_SAE.pt"):
        p = os.path.join(EMB_ROOT, name)
        if os.path.isfile(p):
            return p
    return None

def main():
    # 1) features to use
    if not os.path.isfile(ranking_csv):
        raise FileNotFoundError(ranking_csv)
    rank_df   = pd.read_csv(ranking_csv).sort_values("rank")
    feats_all = rank_df["feature"].astype(int).tolist()
    if len(feats_all) < num_features:
        raise RuntimeError(f"Ranking only has {len(feats_all)} features; need {num_features}")
    top_feats = feats_all[:num_features]
    print(f"✔ Using top {num_features} features (max index {max(top_feats)})")

    # 2) union of all folds
    allowed = set()
    for k in range(10):
        fp = os.path.join(FOLDS_ROOT, f"fold_{k}.txt")
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                allowed.update(fh.read().split())
    if not allowed:
        raise RuntimeError(f"No entries under {FOLDS_ROOT}")
    print(f"✔ Combined {len(allowed)} entries from folds")

    # 3) labels
    pos_df = pd.read_csv(POS_CSV).assign(label=1)
    neg_df = pd.read_csv(NEG_CSV).assign(label=0)
    labels = pd.concat([pos_df, neg_df], ignore_index=True)
    labels["residue_number"] = labels["residue_number"].astype(int)
    labels = labels[labels["Entry"].isin(allowed)].reset_index(drop=True)

    # 4) collect embeddings
    X_rows, y_rows = [], []
    missing = set()
    for _, r in labels.iterrows():
        entry = r["Entry"]
        idx0  = r["residue_number"] - 1
        fp    = find_embedding_path(entry)
        if fp is None:
            missing.add(entry); continue
        arr = torch.load(fp, map_location="cpu").numpy()
        if 0 <= idx0 < arr.shape[0]:
            X_rows.append(arr[idx0, :])
            y_rows.append(int(r["label"]))
    if missing:
        print(f"⚠️ Missing embeddings for {len(missing)} entries (skipped).")

    if not X_rows:
        raise RuntimeError("No training rows collected.")
    X_all = np.vstack(X_rows)
    y_all = np.asarray(y_rows, dtype=int)
    print(f"✔ X={X_all.shape}, positives={y_all.sum()}, negatives={(y_all==0).sum()}")

    # 5) slice, scale, balanced LR (+ optional calibration)
    X_train = X_all[:, top_feats]
    base_lr = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        C=0.25,
        solver="lbfgs"
    )
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", base_lr),
    ])
    if USE_CALIBRATION:
        # Calibrate probabilities with CV – avoids saturation
        clf = CalibratedClassifierCV(pipe, cv=CALIB_CV, method="sigmoid")
    else:
        clf = pipe

    clf.fit(X_train, y_all)
    print("✔ Trained final balanced+scaled LR (calibrated: %s)" % USE_CALIBRATION)

    # 6) save model + feature list
    joblib.dump(clf, model_out)
    pd.DataFrame({"feature": top_feats}).to_csv(features_out_csv, index=False)
    print(f"✔ Saved model → {model_out}")
    print(f"✔ Saved features → {features_out_csv}")

if __name__ == "__main__":
    main()
