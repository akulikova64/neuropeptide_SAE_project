#!/usr/bin/env python3
"""
split_folds_pos_neg.py

• For each fold 0 – 8:
    training_data/group_2_positive_toxin_neuro_train_fold{n}.csv
    training_data/group_2_negative_toxin_neuro_train_fold{n}.csv

• For fold 9:
    test_data/group_2_positive_toxin_neuro_test.csv
    test_data/group_2_negative_toxin_neuro_test.csv
"""

from pathlib import Path
import pandas as pd

# ─── paths & constants ─────────────────────────────────────────────────────
ROOT        = Path("/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data")
FOLD_DIR    = ROOT / "folds"

POS_CSV     = ROOT / "group_2_positive_toxin_neuro.csv"
NEG_CSV     = ROOT / "group_2_negative_toxin_neuro.csv"

TRAIN_DIR   = ROOT / "training_data"
TEST_DIR    = ROOT / "test_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True,  exist_ok=True)

TRAIN_FOLDS = range(0, 9)   # folds 0…8
TEST_FOLD   = 9             # fold 9

# ─── load full tables once ─────────────────────────────────────────────────
pos_df = pd.read_csv(POS_CSV, dtype={"Entry": str})
neg_df = pd.read_csv(NEG_CSV, dtype={"Entry": str})

# ─── training folds (0‑8)  — same logic for pos & neg ─────────────────────
for fold in TRAIN_FOLDS:
    entries = set((FOLD_DIR / f"fold_{fold}.txt").read_text().split())

    pos_fold = pos_df[pos_df["Entry"].isin(entries)]
    neg_fold = neg_df[neg_df["Entry"].isin(entries)]

    pos_fold.to_csv(TRAIN_DIR / f"group_2_positive_toxin_neuro_train_fold{fold}.csv",
                    index=False)
    neg_fold.to_csv(TRAIN_DIR / f"group_2_negative_toxin_neuro_train_fold{fold}.csv",
                    index=False)

    print(f"TRAIN fold {fold}:  +pos {len(pos_fold):,}   -neg {len(neg_fold):,}")

# ─── test fold (9) — same logic for pos & neg ──────────────────────────────
test_entries = set((FOLD_DIR / f"fold_{TEST_FOLD}.txt").read_text().split())

pos_test = pos_df[pos_df["Entry"].isin(test_entries)]
neg_test = neg_df[neg_df["Entry"].isin(test_entries)]

pos_test.to_csv(TEST_DIR / "group_2_positive_toxin_neuro_test.csv", index=False)
neg_test.to_csv(TEST_DIR / "group_2_negative_toxin_neuro_test.csv", index=False)

print(f"TEST  fold 9:  +pos {len(pos_test):,}   -neg {len(neg_test):,}")

