#!/usr/bin/env python3
import os
import csv
import pandas as pd

TSV_IN = "../../data/fig_6_zebrafish_secretome/secratome_annotations_zebrafish.tsv"
CSV_OUT = "../../data/fig_6_zebrafish_secretome/secratome_annotations_zebrafish_clean.csv"

tsv_abs = os.path.abspath(TSV_IN)
csv_abs = os.path.abspath(CSV_OUT)

if not os.path.exists(tsv_abs):
    raise FileNotFoundError(f"TSV not found: {tsv_abs}")

# Read TSV as strings; keep empty cells as empty strings
df = pd.read_csv(
    tsv_abs,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

# Optional safety check (matches your header)
expected_cols = 26
if df.shape[1] != expected_cols:
    raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")

os.makedirs(os.path.dirname(csv_abs), exist_ok=True)

# Write CSV like your cleaned mouse file:
# - quote only when needed
# - escape embedded " by doubling it
df.to_csv(
    csv_abs,
    index=False,
    quoting=csv.QUOTE_MINIMAL,
    quotechar='"',
    doublequote=True,
    lineterminator="\n",   # or "\r\n" if you want Windows-style
)

print(f"✅ Wrote: {csv_abs}")
print(f"   Rows: {df.shape[0]}  Cols: {df.shape[1]}")