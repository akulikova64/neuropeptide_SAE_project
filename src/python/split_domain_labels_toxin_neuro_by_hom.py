#!/usr/bin/env python3
import os
import pandas as pd

# here I was spliting the domain csv with positive labels and the domain they belong to 
# according the the homology sorting I previously did

# ─── Configuration ──────────────────────────────────────────────────────────
HOMOLOGIES   = [40, 50, 60, 70, 80, 90]
INPUT_ROOT   = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data"
DOMAIN_CSV   = os.path.join(INPUT_ROOT, "group_2_positive_toxin_neuro_domain.csv")
TRAIN_ROOT   = os.path.join(INPUT_ROOT, "training_data")

# ─── Load the full domain table ─────────────────────────────────────────────
domain_df = pd.read_csv(DOMAIN_CSV, dtype={"Entry": str})

for hom in HOMOLOGIES:
    # ─── Load training entries list for this homology ────────────────────────
    train_entries_fp = os.path.join(
        TRAIN_ROOT,
        f"homology_group{hom}",
        "training_entry_names.txt"
    )
    if not os.path.isfile(train_entries_fp):
        print(f"⚠️  Missing entries.txt for homology {hom}: {train_entries_fp}")
        continue

    with open(train_entries_fp, "r") as f:
        train_entries = f.read().split()

    # ─── Filter domain_df to only those Entries in the train set ───────────
    df_hom = domain_df[domain_df["Entry"].isin(train_entries)].copy()

    # ─── Prepare output directory & filename ────────────────────────────────
    out_dir = os.path.join(TRAIN_ROOT, f"homology_group{hom}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(
        out_dir,
        f"group_2_positive_neuro_toxin_domain_{hom}.csv"
    )

    # ─── Save filtered domain table ────────────────────────────────────────
    df_hom.to_csv(out_csv, index=False)
    print(f"✅ Homology {hom}: wrote {len(df_hom)} rows to {out_csv}")
