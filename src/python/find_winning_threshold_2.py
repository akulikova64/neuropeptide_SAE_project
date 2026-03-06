#!/usr/bin/env python3
# Note this script is from when I ran my concept_1 on all the layers of the SAE
# ---------------------------------------------------------------------------
#  For every layer directory, pick the threshold that maximises F1 for each
#  feature and save (threshold, F1, accuracy) in a layer-specific CSV.
# ---------------------------------------------------------------------------
from pathlib import Path
import re
import pandas as pd

# ─── Root folder that contains layer_* subfolders ─────────────────────────────
base_dir = Path("/Volumes") / "T7 Shield" / "group_1_F1_scores"

# ─── Regex helper to recover threshold from file name -------------------------
thr_re = re.compile(r"thr_(\d+)(?:_(\d+))?\.csv$")   # captures 0.csv or 0_15.csv

for layer_dir in sorted(base_dir.glob("layer_*")):
    if not layer_dir.is_dir():
        continue
    layer_num = int(layer_dir.name.split("_")[1])
    print(f"\n▶︎ Layer {layer_num}")

    # ── collect all metric rows for this layer ───────────────────────────────
    layer_records = []
    for csv_path in layer_dir.glob("metrics_thr_*.csv"):
        m = thr_re.search(csv_path.name)
        if not m:
            print(f"  ⚠️  skipping {csv_path.name} (unrecognised)")
            continue
        major, minor = m.group(1), m.group(2) or "0"
        thr          = float(f"{major}.{minor}")

        df = pd.read_csv(csv_path,
                         usecols=["feature", "F1_score", "accuracy"])
        df["threshold"] = thr
        layer_records.append(df)

    if not layer_records:
        print(f"  ⚠️  no metric CSVs found in {layer_dir}")
        continue

    layer_df = pd.concat(layer_records, ignore_index=True)

    # ── choose the best threshold per feature --------------------------------
    idx_best = layer_df.groupby("feature")["F1_score"].idxmax()
    best_df  = (layer_df
                .loc[idx_best, ["feature", "threshold", "F1_score", "accuracy"]]
                .rename(columns={
                    "threshold": "winning_threshold",
                    "F1_score":  "winning_F1_score",
                    "accuracy":  "winning_accuracy",
                })
                .sort_values("feature")
                .reset_index(drop=True))

    # ── save in the same layer folder ----------------------------------------
    out_path = layer_dir / f"winning_thresholds_layer_{layer_num}.csv"
    best_df.to_csv(out_path, index=False)
    print(f"  ✓ wrote {len(best_df)} features → {out_path}")

print("\n✅ All layers processed.")
