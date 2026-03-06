#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import joblib
import torch
import numpy as np
import pandas as pd
from glob import glob
from Bio import SeqIO

# ─── Parameters ─────────────────────────────────────────────────────────────
homology_training = 90
number_features   = 600

# ─── Paths ──────────────────────────────────────────────────────────────────
model_path = (
    f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"classifier_results/logistic_regression_models/hom{homology_training}/"
    f"logreg_hom{homology_training}_{number_features}.joblib"
)

emb_dir = "../../data/concept_groups/embeddings/group_2/examples"
fasta_file = "../../data/novo_examples.fasta"

feature_rank_path = (
    f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"input_data/training_data/homology_group{homology_training}/"
    f"feature_ranking_hybrid_hom{homology_training}.csv"
)

output_csv = "../../data/linear_regression_data/example_predictions_logreg_hom90_600_new.csv"

# ─── Load trained logistic regression model ─────────────────────────────────
clf = joblib.load(model_path)

# ─── Load top N features based on rank ──────────────────────────────────────
feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
feature_indices = feat_df["feature"].tolist()[:number_features]

# ─── Load sequences from FASTA ──────────────────────────────────────────────
seq_dict = {}
for rec in SeqIO.parse(fasta_file, "fasta"):
    parts = rec.id.split("|")
    accession = parts[1] if len(parts) >= 2 else rec.id
    seq_dict[accession] = str(rec.seq)

# ─── Run inference ──────────────────────────────────────────────────────────
records = []
for emb_file in sorted(glob(os.path.join(emb_dir, "*.pt"))):
    basename = os.path.basename(emb_file)
    header = basename.replace("_original_SAE.pt", "")
    entry = header.split("|")[1]

    tensor = torch.load(emb_file, map_location="cpu")
    arr = tensor.numpy()

    if arr.shape[1] < len(feature_indices):
        raise ValueError(f"{entry}: only {arr.shape[1]} features, need {len(feature_indices)}")

    X = arr[:, feature_indices]
    probs = clf.predict_proba(X)[:, 1]

    seq = seq_dict.get(entry)
    if seq is None:
        raise KeyError(f"Sequence for {entry} not found in {fasta_file}")

    for pos, (res, p) in enumerate(zip(seq, probs), start=1):
        records.append({
            "Entry": entry,
            "position": pos,
            "residue": res,
            "probability_cleavage_site": p
        })

# ─── Save predictions to CSV ────────────────────────────────────────────────
df = pd.DataFrame(records)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"Saved predictions for {len(df)} residues to {output_csv}")
