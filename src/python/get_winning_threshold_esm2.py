# this code will produce a ranking file with all positive and negative data (all folds)

#!/usr/bin/env python3
"""
Pick the winning (best-F1) threshold per feature from an F1 table.

Input:  CSV with columns [feature, threshold, F1_score, ...]
Output: CSV with one row per feature:
        [feature, winning_threshold, winning_F1_score]
"""

import os
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Pick best threshold per feature from F1 table."
    )
    ap.add_argument(
        "--in",
        dest="in_csv",
        required=True,
        help="Path to F1_seqmax_all_thresholds_filtered.csv",
    )
    ap.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Output CSV path (default: same folder, winning_thresholds.csv)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Print top-K features by winning F1 (default: 20)",
    )
    args = ap.parse_args()

    in_csv = os.path.abspath(args.in_csv)
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    if args.out_csv is None:
        out_csv = os.path.join(
            os.path.dirname(in_csv),
            "winning_thresholds_esm2_l18.csv",
        )
    else:
        out_csv = os.path.abspath(args.out_csv)

    print(f"[i] Reading F1 table: {in_csv}")
    df = pd.read_csv(in_csv)

    # ── Sanity checks ────────────────────────────────────────────────
    required_cols = {"feature", "threshold", "F1_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure proper dtypes
    df["feature"] = df["feature"].astype(int)
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["F1_score"] = pd.to_numeric(df["F1_score"], errors="coerce")

    # Drop invalid rows
    n_before = len(df)
    df = df.dropna(subset=["feature", "threshold", "F1_score"])
    n_after = len(df)
    if n_after < n_before:
        print(f"[i] Dropped {n_before - n_after} rows with NA values")

    # ── Pick winning threshold per feature ───────────────────────────
    # idxmax → first occurrence if ties (deterministic)
    best_idx = df.groupby("feature")["F1_score"].idxmax()
    best_df = df.loc[best_idx, ["feature", "threshold", "F1_score"]].copy()

    best_df.rename(
        columns={
            "threshold": "winning_threshold",
            "F1_score": "winning_F1_score",
        },
        inplace=True,
    )

    # Sort by feature index for stable downstream use
    best_df = best_df.sort_values("feature").reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_df.to_csv(out_csv, index=False)

    print(f"[✓] Wrote winning-threshold table → {out_csv}")
    print(f"[i] Features covered: {best_df['feature'].nunique():,}")

    # ── Report top-K features ────────────────────────────────────────
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
command python get_winning_threshold_esm2.py \
  --in /Volumes/T7\ Shield/ESM2_F1_scores/ESM2_F1_seqmax_all_thresholds_filtered.csv
'''