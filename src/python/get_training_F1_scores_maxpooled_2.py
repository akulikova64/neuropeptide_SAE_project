#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Compute TP / FP / FN / TN for each SAE feature using max-pooled activations
#  across all sequences in the "positive" and "negative" folders of every layer.
#  Metrics: precision, recall, F1, and accuracy.
# ---------------------------------------------------------------------------

from pathlib import Path
import torch
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
t7_root      = Path("/Volumes") / "T7 Shield"
layers       = [1, 9, 18, 24, 30, 33]          # SAE layers that I have embeddings for
thresholds   = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

# helper ───────────────────────────────────────────────────────────────────────
def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0

# ─── Iterate over layers ──────────────────────────────────────────────────────
for layer in layers:
    print(f"\n▶︎ Processing layer {layer}")

    # Input dirs with .pt tensors
    layer_dir     = t7_root / "group_1_embeddings" / f"layer_{layer}"
    pos_dir       = layer_dir / "positive"
    neg_dir       = layer_dir / "negative"

    # Determine #features from any sample tensor
    sample_pt   = next(pos_dir.glob("*.pt"))
    num_feat    = torch.load(sample_pt, map_location="cpu").shape[1]
    print(f"  • detected {num_feat} SAE features")

    # Prepare output folder
    out_layer_dir = t7_root / "group_1_F1_scores" / f"layer_{layer}"
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    # Load and cache max-pooled activations to use for each threshold -------------------------
    cache = {"positive": [], "negative": []}  # list of 1-D tensors per dataset

    for dataset, dataset_directory in [("positive", pos_dir), ("negative", neg_dir)]:

        for pt_file in dataset_directory.glob("*.pt"):
            if pt_file.name.startswith("._"):
                continue           # <-- ignoring the resource-fork file
            T = torch.load(pt_file, map_location="cpu")  # shape (residues, features)
            cache[dataset].append(T.max(dim=0).values)   # dim = 0 is residues

    # ── Loop over thresholds ─────────────────────────────────────────────────
    for threshold in thresholds:
        print(f"    • threshold {threshold}")
        # counts per feature: TP, FP, FN, TN
        counts = {
            feature: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            for feature in range(num_feat)
        }

        # positives
        for activation in cache["positive"]:
            pred = (activation > threshold)
            for feature, is_pos in enumerate(pred.tolist()):
                if is_pos:
                    counts[feature]["TP"] += 1
                else:
                    counts[feature]["FN"] += 1

        # negatives
        for activation in cache["negative"]:
            pred = (activation > threshold)
            for feature, is_pos in enumerate(pred.tolist()):
                if is_pos:
                    counts[feature]["FP"] += 1
                else:
                    counts[feature]["TN"] += 1

        # ── Compute metrics & collect rows ───────────────────────────────────
        rows = []
        for feature, count in counts.items():
            TP, FP, FN, TN = count["TP"], count["FP"], count["FN"], count["TN"]
            precision = safe_div(TP, TP + FP)
            recall  = safe_div(TP, TP + FN)
            f1   = safe_div(2 * precision * recall, precision + recall)
            accuracy  = safe_div(TP + TN, TP + FP + FN + TN)

            rows.append(
                {
                    "feature":   feature,
                    "threshold": threshold,
                    "precision": precision,
                    "recall":    recall,
                    "F1_score":  f1,
                    "accuracy":  accuracy,
                }
            )

        # Data-frame → CSV ----------------------------------------------------
        df_out = pd.DataFrame(rows).sort_values("feature")
        cols   = ["feature", "threshold", "precision", "recall", "F1_score", "accuracy"]
        df_out = df_out[cols]

        csv_name = f"metrics_thr_{str(threshold).replace('.', '_')}.csv"
        df_out.to_csv(out_layer_dir / csv_name, index=False)
        print(f"      ✓ saved → {out_layer_dir/csv_name}")

print("\n✅ All done.")
