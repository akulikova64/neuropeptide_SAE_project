import os
import torch
import pandas as pd

# extract 

# ─── USER PARAMETERS ────────────────────────────────────────────────────
embeddings_dir = "/Volumes/T7 Shield/layer_18_secratome_embeddings"
feature_idx    = 4813   # change to the feature you want
output_csv     = "../../data/neuropep_features/feature_{}_activations_ALL.csv".format(feature_idx)

# ─── COLLECT ACTIVATIONS ────────────────────────────────────────────────
records = []
missing = []

for fn in os.listdir(embeddings_dir):
    if not fn.endswith(".pt"):
        continue
    # extract entry ID before the first underscore
    entry = fn.split("_", 1)[0]
    #entry = fn.split("|")[1]
    fpath = os.path.join(embeddings_dir, fn)
    
    try:
        tensor = torch.load(fpath, map_location="cpu")  # shape: (seq_len, num_features)
    except Exception as e:
        missing.append(entry)
        continue
    
    seq_len, num_features = tensor.shape
    if feature_idx < 0 or feature_idx >= num_features:
        raise ValueError(f"feature_idx {feature_idx} out of range (0–{num_features-1})")

    # extract this feature's activation at every position
    for pos in range(seq_len):
        records.append({
            "Entry":     entry,
            "position":  pos + 1,            # 1-based position
            "activation": float(tensor[pos, feature_idx].item())
        })

if missing:
    print(f"Warning: failed to load embeddings for entries: {missing}")

# ─── SAVE TO CSV ─────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_out = pd.DataFrame.from_records(records, columns=["Entry", "position", "activation"])
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved {len(records)} rows to {output_csv}")