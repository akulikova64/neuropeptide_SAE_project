#!/usr/bin/env python3
import os
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Pick best threshold per feature from F1 table.")
    ap.add_argument("--in", dest="in_csv", required=True,
                    help="Path to F1_seqmax_all_thresholds.csv")
    ap.add_argument("--out", dest="out_csv", default=None,
                    help="Output CSV path (default: same folder, best_threshold_per_feature.csv)")
    ap.add_argument("--topk", type=int, default=20, help="Print top-K features (default: 10)")
    args = ap.parse_args()

    in_csv = os.path.abspath(args.in_csv)
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(in_csv), "winning_thresholds_l18.csv")

    print(f"[i] Reading: {in_csv}")
    df = pd.read_csv(in_csv)

    # Basic sanity checks
    required_cols = {"feature", "threshold", "F1_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Ensure types are sensible
    df["feature"] = df["feature"].astype(int)
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["F1_score"] = pd.to_numeric(df["F1_score"], errors="coerce")

    # Drop any rows with NA in critical fields
    df = df.dropna(subset=["feature", "threshold", "F1_score"])

    # Pick row index per feature where F1 is maximal
    # (If ties, idxmax picks the first occurrence.)
    best_idx = df.groupby("feature")["F1_score"].idxmax()
    best_df = df.loc[best_idx, ["feature", "threshold", "F1_score"]].copy()
    best_df.rename(columns={
        "threshold": "winning_threshold",
        "F1_score":  "winning_F1_score"
    }, inplace=True)

    # Sort by feature (stable), then save
    best_df = best_df.sort_values(["feature"]).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_df.to_csv(out_csv, index=False)
    print(f"[✓] Wrote best-per-feature CSV → {out_csv}")
    print(f"[i] Features covered: {best_df['feature'].nunique()}")

    # Print top-K by winning F1 (desc)
    topk = min(args.topk, len(best_df))
    top = best_df.sort_values("winning_F1_score", ascending=False).head(topk)
    print(f"\nTop {topk} features by winning F1:")
    for _, row in top.iterrows():
        feat = int(row["feature"])
        thr  = float(row["winning_threshold"])
        f1   = float(row["winning_F1_score"])
        print(f"  feature {feat:5d}  F1={f1:.4f}  threshold={thr:g}")

if __name__ == "__main__":
    main()



'''
command python get_winning_threshold_l18.py \
  --in "/Volumes/T7 Shield/layer_18_F1_scores/F1_seqmax_all_thresholds_filtered.csv"
'''  