#!/usr/bin/env python3
import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from Bio import SeqIO
import torch.nn as nn

# ─── Model factory ─────────────────────────────────────────────────────────
def make_model(num_features, dropout_rate):
    # hidden layer size scales with input but capped between 5 and 50
    h1 = min(max(5, num_features // 5), 50)
    return nn.Sequential(
        nn.Linear(num_features, h1),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(h1, 1),
        nn.Sigmoid()
    )

# ─── Configuration ─────────────────────────────────────────────────────────
num_features = 300
dropout_rate = 0.3

model_path  = "/Volumes/T7 Shield/nn_models/90/nn_hom90_500.pt"
emb_dir     = "../../data/concept_groups/embeddings/group_2/examples"
fasta_file  = "../../data/novo_examples.fasta"
output_csv  = "../../data/vanilla_nn_results/example_predictions_hom90_500.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load your trained model ────────────────────────────────────────────────
model = make_model(num_features, dropout_rate).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ─── Load sequences from FASTA ──────────────────────────────────────────────
# Map UniProt accession -> sequence string
seq_dict = {}
for rec in SeqIO.parse(fasta_file, "fasta"):
    # rec.id might be like "sp|A0A0U1RR37|CA232_HUMAN"
    parts = rec.id.split("|")
    accession = parts[1] if len(parts) >= 2 else rec.id
    seq_dict[accession] = str(rec.seq)

# ─── Inference over embeddings ──────────────────────────────────────────────
records = []
for emb_file in sorted(glob(os.path.join(emb_dir, "*.pt"))):
    basename = os.path.basename(emb_file)
    header   = basename.replace("_original_SAE.pt", "")
    # header e.g. "sp|A0A0U1RR37|CA232_HUMAN"
    entry    = header.split("|")[1]

    # load embedding (n_positions × total_features)
    tensor = torch.load(emb_file, map_location="cpu")
    arr    = tensor.numpy()
    if arr.shape[1] < num_features:
        raise ValueError(f"{entry}: only {arr.shape[1]} features, need {num_features}")

    X = torch.from_numpy(arr[:, :num_features]).float().to(device)
    with torch.no_grad():
        probs = model(X).cpu().numpy().flatten()

    seq = seq_dict.get(entry)
    if seq is None:
        raise KeyError(f"Sequence for {entry} not found in FASTA {fasta_file}")

    for pos, p in enumerate(probs, start=1):
        residue = seq[pos - 1]  # 1-based position
        records.append({
            "Entry": entry,
            "position": pos,
            "residue": residue,
            "probability_cleavage_site": p
        })

# ─── Save to CSV ────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} predictions with residues to {output_csv}")
