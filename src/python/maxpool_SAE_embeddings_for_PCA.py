#!/usr/bin/env python3
"""
Max-pool SAE layer-18 embeddings for selected human secretome entries.

This version assumes:
- .pt files contain ONLY a tensor of shape (L, F)
- Entry ID must be extracted from the filename:
  sp|ACCESSION|NAME_original_SAE.pt
"""

import os
import torch
import pandas as pd
from glob import glob

# ---------------- CONFIG ----------------

EMB_DIR = "/Volumes/T7 Shield/layer_18_secratome_embeddings"
ENTRY_CSV = "../../data/final_human_secretome_entries_1201.csv"
OUT_CSV = "../../data/sae_layer18_secretome_embeddings_maxpooled_1201.csv"

# ---------------- HELPERS ----------------

def load_allowed_entries(path):
    df = pd.read_csv(path)
    if "Entry" not in df.columns:
        raise ValueError("Entry CSV must contain an 'Entry' column")
    return set(df["Entry"].astype(str))

def extract_accession_from_filename(fname):
    """
    Extract UniProt accession from filenames like:
    sp|A0A075B6H7|KV37_HUMAN_original_SAE.pt
    """
    base = os.path.basename(fname)
    base = base.replace("_original_SAE.pt", "")
    parts = base.split("|")
    if len(parts) >= 3:
        return parts[1]
    raise ValueError(f"Cannot extract accession from filename: {fname}")

# ---------------- MAIN ----------------

def main():
    print("[i] Loading allowed Entry list")
    allowed_entries = load_allowed_entries(ENTRY_CSV)
    print(f"[i] Allowed entries: {len(allowed_entries)}")

    pt_files = glob(os.path.join(EMB_DIR, "*.pt"))
    print(f"[i] Found {len(pt_files)} .pt files on disk")

    # Map accession -> path
    entry_to_path = {}
    for p in pt_files:
        try:
            acc = extract_accession_from_filename(p)
        except Exception:
            continue
        if acc in allowed_entries:
            entry_to_path[acc] = p

    print(f"[i] Matched {len(entry_to_path)}/{len(allowed_entries)} entries")

    if len(entry_to_path) == 0:
        raise RuntimeError("No embeddings matched the allowed Entry list")

    rows = []
    feature_dim = None

    for i, (acc, p) in enumerate(entry_to_path.items(), 1):
        emb = torch.load(p, map_location="cpu")

        if not torch.is_tensor(emb):
            raise ValueError(f"{p} does not contain a tensor")

        if emb.ndim != 2:
            raise ValueError(f"{acc}: embedding is not 2D")

        # Max-pool across residues
        v = emb.max(dim=0).values.cpu().numpy()

        if feature_dim is None:
            feature_dim = v.shape[0]
            print(f"[i] Feature dimension: {feature_dim}")

        rows.append([acc] + v.tolist())

        if i % 50 == 0 or i == len(entry_to_path):
            print(f"[i] Processed {i}/{len(entry_to_path)}")

    print(f"[i] Writing output CSV: {OUT_CSV}")

    colnames = ["Entry"] + [f"e_{i+1}" for i in range(feature_dim)]
    out_df = pd.DataFrame(rows, columns=colnames)
    out_df.to_csv(OUT_CSV, index=False)

    print("== DONE ===============================")
    print(f"Rows written: {len(rows)}")
    print(f"Feature dim:  {feature_dim}")
    print("======================================")

if __name__ == "__main__":
    main()
