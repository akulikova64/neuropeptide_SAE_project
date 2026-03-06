#!/usr/bin/env python3
import os
import time
import requests
import pandas as pd
from io import StringIO

# ─── Paths ────────────────────────────────────────────────────────────────
INPUT_CSV  = "../../data/test_dataset_stats/human_mature_pep.csv"
OUTPUT_CSV = "../../data/test_dataset_stats/human_mature_pep_annotations.csv"

# ─── UniProt fields to retrieve ───────────────────────────────────────────
FIELDS = [
    "accession","id","protein_name","organism_name",
    "sequence","length","cc_ptm","ft_signal",
    "ft_propep","ft_peptide","go_f","go_p","protein_families"
]
FIELD_STR  = ",".join(FIELDS)
STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

# ─── Prepare output directory ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ─── Load Uniprot IDs ─────────────────────────────────────────────────────
df_input = pd.read_csv(INPUT_CSV, dtype=str)
if "Uniprot" not in df_input.columns:
    raise KeyError("Input CSV must have a 'Uniprot' column")
ids = df_input["Uniprot"].dropna().unique().tolist()

if not ids:
    raise ValueError("No Uniprot IDs found in input")

# ─── Build a single large query string ────────────────────────────────────
#   accession:ID1 OR accession:ID2 OR ...
query = " OR ".join(f"accession:{u}" for u in ids)

# ─── Fetch all annotations in one request ─────────────────────────────────
params = {
    "query":  query,
    "format": "tsv",
    "fields": FIELD_STR
}
print(f"Requesting {len(ids)} entries from UniProt…")
resp = requests.get(STREAM_URL, params=params)
resp.raise_for_status()

text = resp.text.strip()
if not text:
    raise RuntimeError("Empty response from UniProt")

# ─── Parse the TSV into a DataFrame ───────────────────────────────────────
# first line = header, following lines = one per accession (for those found)
df_out = pd.read_csv(StringIO(text), sep="\t", dtype=str)

# ─── Write to CSV ─────────────────────────────────────────────────────────
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Written {len(df_out)} annotated entries to {OUTPUT_CSV}")
