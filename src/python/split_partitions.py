#!/usr/bin/env python3
"""
split_partitions.py
Create one TXT file (0.txt … 9.txt) per cluster with AC values space‑separated.
"""

import pandas as pd
from pathlib import Path

# -------- paths -------------------------------------------------------------
csv_path   = Path("/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/neuro_toxin_data_all_partitions_40.csv")
output_dir = Path("/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_40")        
output_dir.mkdir(exist_ok=True)

# -------- read CSV ----------------------------------------------------------
df = pd.read_csv(csv_path)

# Ensure the cluster column is integer 0–9 (it may be read as float)
df["cluster"] = df["cluster"].astype(int)

# -------- write one file per cluster ----------------------------------------
for cl in range(10):                       # clusters 0 … 9
    ac_list = df.loc[df["cluster"] == cl, "AC"].tolist()
    out_file = output_dir / f"fold_{cl}.txt"
    with out_file.open("w") as f:
        f.write(" ".join(ac_list))

    print(f"Wrote {len(ac_list):>4} entries → {out_file}")

print("Done.")
