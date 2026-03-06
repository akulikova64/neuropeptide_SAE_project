#!/usr/bin/env python3
"""
Make a FASTA from a CSV with columns:
  - smorf.id   -> FASTA header (after '>')
  - input.Pep  -> amino-acid sequence

Filters:
  - must start with 'M'
  - must be ≤ 3000 residues
"""

import os
import sys
import csv

def bump_field_limit():
    import sys, csv
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
            if max_int < 131072:
                csv.field_size_limit(10_000_000)
                break

def main():
    csv_path = "../../data/20250611_NN_smORFpipe_full_data.csv"
    out_path = "../../data/novo_smORFpipe_full_data.fasta"

    bump_field_limit()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_with_seq = 0
    kept = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f, \
         open(out_path, "w", encoding="utf-8", newline="") as out:
        rdr = csv.DictReader(f)
        if not {"smorf.id", "input.Pep"} <= set(rdr.fieldnames or []):
            raise ValueError("Missing required columns: smorf.id, input.Pep")

        for row in rdr:
            sid = (row.get("smorf.id") or "").strip()
            pep = (row.get("input.Pep") or "").strip().replace(" ", "").replace("\n", "").upper()

            if not pep:
                continue

            total_with_seq += 1
            # Filter by start and length
            if pep.startswith("M") and len(pep) <= 3000:
                kept += 1
                out.write(f">{sid}\n{pep}\n")

    print(f"Total sequences with non-empty 'input.Pep': {total_with_seq}")
    print(f"Kept (start with 'M' and length ≤ 3000):   {kept}")
    print(f"FASTA written to:                           {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
