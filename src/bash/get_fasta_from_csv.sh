#!/usr/bin/env bash
set -euo pipefail

CSV="../../data/20250611_NN_smORFpipe_full_data.csv"
OUT="../../data/20250611_NN_smORFpipe_full_data.fasta"

python3 - <<'PY' "$CSV" "$OUT"
import sys, csv, io, os

csv_path, out_path = sys.argv[1], sys.argv[2]

total = 0
kept  = 0

# Ensure output dir exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with io.open(csv_path, 'r', encoding='utf-8', newline='') as f, \
     io.open(out_path, 'w', encoding='utf-8', newline='') as out:
    rdr = csv.DictReader(f)
    # Normalize header keys to match exactly
    need_cols = {'smorf.id', 'input.Pep'}
    missing = need_cols - set(rdr.fieldnames or [])
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}. "
                         f"Found: {rdr.fieldnames}")

    for row in rdr:
        total += 1
        sid = (row.get('smorf.id') or '').strip()
        pep = (row.get('input.Pep') or '').strip()

        if not pep:
            continue
        # Filter: sequence must start with 'M'
        if pep and pep[0].upper() == 'M':
            kept += 1
            out.write(f">{sid}\n{pep}\n")

print(f"Total sequences in CSV: {total}")
print(f"Kept (start with M):    {kept}")
print(f"FASTA written to:       {out_path}")
PY
