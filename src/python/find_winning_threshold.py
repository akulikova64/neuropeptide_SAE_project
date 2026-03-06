import os
import re
import pandas as pd
import glob

test_folds = list(range(10))    
percent_identity = 40
run = 2

for test_fold in test_folds:
    base_dir    = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/excluded_fold_{test_fold}/"
    output_file = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/excluded_fold_{test_fold}/winning_thresholds_excluded_fold_{test_fold}.csv"
    os.makedirs(base_dir, exist_ok=True)

    records = []

    # 1) Read each CSV in base_dir
    # only grab “real” .csv files (no dot-files)
    for path in glob.glob(os.path.join(base_dir, "*.csv")):
        fn = os.path.basename(path)
        if fn.startswith('.'):
            continue

        # Expect filenames like "F1_scores_threshold_0_15.csv"
        m = re.search(r"threshold_(\d+)_(\d+)\.csv$", fn)
        if not m:
            # fallback: try threshold_0.csv or threshold_0_5.csv style
            m2 = re.search(r"threshold_(\d+)\.csv$", fn)
            if not m2:
                print(f"Skipping unrecognized filename format: {fn}")
                continue
            thr = float(m2.group(1))
        else:
            major, minor = m.groups()
            thr = float(f"{major}.{minor}")

        path = os.path.join(base_dir, fn)
        print(f"Loading {fn}  → threshold = {thr}")
        df = pd.read_csv(path, usecols=["feature", "F1_score"])
        df["threshold"] = thr
        records.append(df)

    if not records:
        raise RuntimeError(f"No CSVs found or recognized under {base_dir}")

    # 2) Concatenate all thresholds
    all_df = pd.concat(records, ignore_index=True)
    print(f"Combined {len(all_df)} rows from {len(records)} threshold files.")

    # 3) For each feature, pick the row (threshold) with max F1_score
    best_idx = all_df.groupby("feature")["F1_score"].idxmax()
    best_df  = all_df.loc[best_idx, ["feature", "threshold", "F1_score"]].copy()
    best_df.rename(columns={
        "threshold":        "winning_threshold",
        "F1_score":         "winning_F1_score"
    }, inplace=True)

    # 4) Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    best_df.to_csv(output_file, index=False)
    print(f"✅ Wrote {len(best_df)} features → best thresholds in {output_file}")