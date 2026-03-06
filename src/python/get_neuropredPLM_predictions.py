#!/usr/bin/env python3

import torch
import argparse as _argparse
import pandas as pd
from pathlib import Path
from Bio import SeqIO

# PyTorch 2.6+ safety fix
torch.serialization.add_safe_globals([_argparse.Namespace])

from NeuroPredPLM.predict import predict


# ---------------- CONFIG ----------------

FASTA_IN = "../../data/final_human_secretome_entries_1201.fasta"
OUT_CSV  = "../../data/neuropredPLM_predictions_1201.csv"
BATCH_SIZE = 8   # safe on macOS


# ---------------- HELPERS ----------------

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ---------------- MAIN ----------------

def main():
    fasta_path = Path(FASTA_IN)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    print(f"[i] Reading FASTA: {fasta_path}")

    peptide_list = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        entry = rec.id.split()[0]
        seq = str(rec.seq)
        peptide_list.append((entry, seq))

    total = len(peptide_list)
    print(f"[i] Loaded {total} sequences")

    rows = []
    processed = 0

    print(f"[i] Running NeuroPredPLM in batches of {BATCH_SIZE}")

    for batch in batched(peptide_list, BATCH_SIZE):
        out = predict(batch)

        for entry, (pred_class, attn) in out.items():
            max_activation  = float(attn.max())
            mean_activation = float(attn.mean())

            rows.append((
                entry,
                int(pred_class),
                max_activation,
                mean_activation
            ))

        processed += len(batch)
        print(f"[i] Processed {processed}/{total}")

        # free memory
        del out
        torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows,
        columns=[
            "Entry",
            "predicted_class",
            "max_activation",
            "mean_activation"
        ]
    )

    df.to_csv(OUT_CSV, index=False)

    print("== DONE ===============================")
    print(f"Rows written: {len(df)}")
    print(f"Output file:  {OUT_CSV}")
    print("=======================================")

if __name__ == "__main__":
    main()
