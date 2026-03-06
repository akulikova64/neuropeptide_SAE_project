#!/usr/bin/env python3
import os
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40
number_of_top_features = 2000
run = 6

for test_fold in test_folds:
    # ─── Paths ───────────────────────────────────────────────────────────────
    base_dir    = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
        f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/"
        f"excluded_fold_{test_fold}"
    )
    input_file  = os.path.join(
        base_dir,
        f"winning_thresholds_excluded_fold_{test_fold}.csv"
    )
    output_file = os.path.join(
        base_dir,
        f"feature_ranking_excluded_fold_{test_fold}.csv"
    )

    if not os.path.isfile(input_file):
        print(f"⚠️  Skipping fold {test_fold}: {input_file} not found")
        continue

    # ─── Load F1 results ────────────────────────────────────────────────────
    in_df = pd.read_csv(input_file)

    # ─── Sort features by descending F1_score and take top 2000 ──────────────
    f1_sorted = in_df.sort_values("winning_F1_score", ascending=False).head(number_of_top_features)

    # ─── Assign ranks (1 = highest F1_score) ─────────────────────────────────
    f1_sorted = f1_sorted.reset_index(drop=True)
    f1_sorted["rank"] = f1_sorted.index + 1

    # ─── Keep only the columns requested ─────────────────────────────────────
    out_df = f1_sorted[["feature", "rank"]]

    # ─── Save to CSV ────────────────────────────────────────────────────────
    out_df.to_csv(output_file, index=False)
    print(f"✔ Fold {test_fold}: saved top {number_of_top_features} features to {output_file}")
