#!/usr/bin/env python3
import os, sys, time, csv
from pathlib import Path
import requests
import pandas as pd

# ── INPUT / OUTPUT ─────────────────────────────────────────────────────────
IN_CSV   = Path("../../data/secretome_filter_out_nocdhit/secretome_filtered_nonenzymes_nocsdhit.csv")
OUT_DIR  = Path("../../data/secretome_filter_out_nocdhit/uniprot_tm_filter")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NON_TM = OUT_DIR / "secretome_nontransmembrane_entries.csv"
OUT_TM     = OUT_DIR / "secretome_transmembrane_entries.csv"
OUT_SUM    = OUT_DIR / "summary.txt"

# UniProt stream endpoint
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# Query: human + Transmembrane keyword (KW-0812).
# (Optionally add "AND keyword:KW-0964" if you want to restrict to UniProt's secreted set.)
UNIPROT_QUERY = "(taxonomy_id:9606) AND (keyword:KW-0812)"

HEADERS = {
    "Accept": "text/tab-separated-values",
    "User-Agent": "tm-filter/1.0 (+you@example.org)"
}

def stream_uniprot_accessions(query: str) -> set[str]:
    """Return set of UniProt accessions for the query."""
    params = {"query": query, "fields": "accession", "format": "tsv"}
    # backoff retries
    for attempt in range(6):
        try:
            r = requests.get(UNIPROT_STREAM, params=params, headers=HEADERS,
                             timeout=120, stream=True)
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        # last response holds the error
        raise SystemExit(f"UniProt request failed (status {r.status_code}): {r.text[:400]}")
    accs = set()
    first = True
    acc_idx = 0
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if first:
            cols = line.rstrip("\n").split("\t")
            acc_idx = cols.index("Entry") if "Entry" in cols else 0
            first = False
            continue
        parts = line.rstrip("\n").split("\t")
        if acc_idx < len(parts):
            accs.add(parts[acc_idx])
    return accs

def main():
    if not IN_CSV.exists():
        sys.exit(f"❌ Input CSV not found: {IN_CSV}")

    # Load your post-enzyme list (single column 'Entry' expected)
    df = pd.read_csv(IN_CSV)
    if "Entry" not in df.columns:
        # try to guess first column
        df.columns = [c.strip() for c in df.columns]
        if "Entry" not in df.columns:
            df.columns = ["Entry"] + [f"col{i}" for i in range(1, len(df.columns))]
    base = set(df["Entry"].astype(str))
    print(f"• Starting with {len(base)} accessions from {IN_CSV}")

    # Fetch UniProt transmembrane set and intersect
    print("• Querying UniProt for transmembrane proteins (KW-0812) …")
    tm_all = stream_uniprot_accessions(UNIPROT_QUERY)
    print(f"  – UniProt TM (human): {len(tm_all)}")

    tm_hit     = sorted(base & tm_all)
    non_tm_hit = sorted(base - tm_all)
    print(f"• Intersections: TM={len(tm_hit)}  non-TM={len(non_tm_hit)}")

    # Write outputs
    pd.DataFrame({"Entry": non_tm_hit}).to_csv(OUT_NON_TM, index=False)
    pd.DataFrame({"Entry": tm_hit}).to_csv(OUT_TM, index=False)

    with open(OUT_SUM, "w") as fh:
        fh.write("UniProt-based transmembrane filtering\n")
        fh.write(f"Input (post-enzyme): {len(base)}\n")
        fh.write(f"Removed as TM:       {len(tm_hit)}\n")
        fh.write(f"Remaining non-TM:    {len(non_tm_hit)}\n")
        fh.write(f"Input CSV:           {IN_CSV}\n")
        fh.write(f"Non-TM CSV:          {OUT_NON_TM}\n")
        fh.write(f"TM CSV:              {OUT_TM}\n")

    print("✅ Done.")
    print(f"  • Non-TM CSV → {OUT_NON_TM}")
    print(f"  • TM CSV     → {OUT_TM}")
    print(f"  • Summary    → {OUT_SUM}")

if __name__ == "__main__":
    main()
