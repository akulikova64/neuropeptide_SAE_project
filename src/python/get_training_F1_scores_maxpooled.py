import os
import os
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────────
input_base = "../../data/concept_groups/embeddings/group_1/max_pooled_training"
positive_dir = os.path.join(input_base, "positive")
negative_dir = os.path.join(input_base, "negative")
output_dir = "../../data/concept_groups/F1_scores/group_1/maxpooled"
os.makedirs(output_dir, exist_ok=True)

thresholds = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

# ─── Helper for safe division ──────────────────────────────────────────────────
def safe_div(num, den):
    return num / den if den else 0.0

# ─── Main loop over thresholds ─────────────────────────────────────────────────
for thr in thresholds:
    print(f"\n🔍 Processing threshold: {thr}")
    feature_counts = {}

    # === POSITIVE files ===
    for file in os.listdir(positive_dir):
        if not file.endswith(".csv"):
            continue
        print(f"Threshold: {thr}, POS file: {file}")
        df = pd.read_csv(os.path.join(positive_dir, file), usecols=["feature", "max_activation"])
        for _, row in df.iterrows():
            feat = int(row["feature"])
            pred_pos = row["max_activation"] > thr
            if feat not in feature_counts:
                feature_counts[feat] = {"TP": 0, "FP": 0, "FN": 0}
            if pred_pos:
                feature_counts[feat]["TP"] += 1
            else:
                feature_counts[feat]["FN"] += 1

    # === NEGATIVE files ===
    for file in os.listdir(negative_dir):
        if not file.endswith(".csv"):
            continue
        print(f"Threshold: {thr}, NEG file: {file}")
        df = pd.read_csv(os.path.join(negative_dir, file), usecols=["feature", "max_activation"])
        for _, row in df.iterrows():
            feat = int(row["feature"])
            pred_pos = row["max_activation"] > thr
            if feat not in feature_counts:
                feature_counts[feat] = {"TP": 0, "FP": 0, "FN": 0}
            if pred_pos:
                feature_counts[feat]["FP"] += 1
            # we do not count TNs

    # === Compute metrics per feature ===
    out_rows = []
    for feat, counts in feature_counts.items():
        TP, FP, FN = counts["TP"], counts["FP"], counts["FN"]
        precision = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        out_rows.append({
            "feature": feat,
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "F1_score": f1
        })

    # === Save to CSV ===
    out_df = pd.DataFrame(out_rows)
    cols_order = ["feature", "threshold", "precision", "recall", "F1_score"]
    out_df = out_df[cols_order]  # reorder to make "feature" first
    out_file = os.path.join(output_dir, f"F1_scores_threshold_{str(thr).replace('.', '_')}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"✅ Saved metrics @ threshold={thr} → {out_file}")
