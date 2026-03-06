#!/usr/bin/env python3
import os
import sys
import csv
import time
import math
import shutil
import subprocess as sp
from pathlib import Path

import requests
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FASTA = Path("../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta")
OUT_DIR     = Path("../../data/secretome_filter_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CD-HIT options
CDHIT_IDENTITY = 0.95
CDHIT_WORDSIZE = 5           # 0.95–1.0 requires -n 5
CDHIT_THREADS  = max(1, os.cpu_count() or 1)

# UniProt options
GO_TERM       = "GO:0003824"  # catalytic activity
UNIPROT_SIZE  = 200           # chunk size for REST calls
UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# MMseqs2 options
MMSEQS_EVALUE = 1e-10
MMSEQS_THREADS = max(1, os.cpu_count() or 1)

# Outputs / intermediates
DEDUP_FASTA   = OUT_DIR / "secretome_dedup.fasta"
CDHIT_CLSTR   = OUT_DIR / "secretome_dedup.fasta.clstr"
ENZ_FASTA     = OUT_DIR / "enzymes_flagged.fasta"
MM_TMP        = OUT_DIR / "mmseqs_tmp"
DB_ENZ        = OUT_DIR / "enz_db"
DB_ALL        = OUT_DIR / "all_db"
DB_RES        = OUT_DIR / "enz_vs_all"
RES_M8        = OUT_DIR / "enz_vs_all.m8"
FINAL_CSV     = OUT_DIR / "secretome_filtered_nonenzymes.csv"

# ── UTILITIES ───────────────────────────────────────────────────────────────
def require_tool(name: str):
    if shutil.which(name) is None:
        sys.exit(f"❌ Required tool '{name}' not found in PATH.")

def fasta_iter_ids(path: Path):
    """Yield (id_token, header_line, sequence) where id_token = first word after '>'."""
    with path.open() as fh:
        hdr = None
        seq = []
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    token = hdr.split()[0]
                    yield token, hdr, "".join(seq)
                hdr = line[1:].strip()
                seq = []
            else:
                seq.append(line.strip())
        if hdr is not None:
            token = hdr.split()[0]
            yield token, hdr, "".join(seq)

def token_to_entry(id_token: str) -> str:
    """
    UniProt FASTA style: sp|P12345|NAME  or tr|A0A...|NAME
    Return accession between '|' when present; else fall back to the whole token.
    """
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token

def run(cmd, **kwargs):
    print("▶", " ".join(cmd))
    sp.run(cmd, check=True, **kwargs)

def uniprot_fetch_go_enzyme_flags(accessions):
    """
    Return the subset of `accessions` that are annotated with catalytic activity (GO:0003824),
    using a single streamed UniProt query:
        (taxonomy_id:9606) AND (keyword:KW-0964) AND (go:0003824)
    Then intersect with our accession list.

    This avoids long OR-lists and the 400 errors you saw.
    """
    headers = {
        "Accept": "text/tab-separated-values",
        "User-Agent": "secretome-filter/1.0 (+your-email@example.com)"
    }

    # Stream ALL human secreted proteins that have GO:0003824
    query  = "(taxonomy_id:9606) AND (keyword:KW-0964) AND (go:0003824)"
    params = {
        "query":  query,
        "fields": "accession",   # 'Entry' column in TSV
        "format": "tsv"
    }

    # robust GET with retry for transient issues
    for attempt in range(4):
        try:
            r = requests.get(UNIPROT_STREAM, params=params, headers=headers,
                             timeout=120, stream=True)
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        r.raise_for_status()  # will throw with the last status

    # Parse TSV stream
    enzyme_all = set()
    first = True
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if first:
            # header line, e.g. "Entry"
            cols = line.rstrip("\n").split("\t")
            acc_idx = cols.index("Entry") if "Entry" in cols else 0
            first = False
            continue
        parts = line.rstrip("\n").split("\t")
        if acc_idx < len(parts):
            enzyme_all.add(parts[acc_idx])

    # Intersect with our accession list
    acc_set = set(accessions)
    enzyme_subset = enzyme_all & acc_set
    return enzyme_subset

# ── PIPELINE ────────────────────────────────────────────────────────────────
def main():
    # 0) sanity checks
    require_tool("cd-hit")
    require_tool("mmseqs")

    if not INPUT_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {INPUT_FASTA}")

    # 1) CD-HIT (95% identity)
    if not DEDUP_FASTA.exists():
        print(f"\n[1/5] Running CD-HIT @ {CDHIT_IDENTITY:.2f} identity …")
        run([
            "cd-hit",
            "-i", str(INPUT_FASTA),
            "-o", str(DEDUP_FASTA),
            "-c", str(CDHIT_IDENTITY),
            "-n", str(CDHIT_WORDSIZE),
            "-T", str(CDHIT_THREADS),
            "-M", "16000"  # change if you want to cap memory (MB)
        ])
    else:
        print(f"\n[1/5] Skipping CD-HIT; {DEDUP_FASTA} exists.")

    # 2) Parse deduplicated FASTA and collect accessions
    print("[2/5] Parsing de-duplicated FASTA and collecting accessions …")
    token_to_seq = {}
    token_to_hdr = {}
    token_to_acc = {}
    for token, hdr, seq in fasta_iter_ids(DEDUP_FASTA):
        token_to_seq[token] = seq
        token_to_hdr[token] = hdr
        token_to_acc[token] = token_to_entry(token)
    all_tokens = list(token_to_seq.keys())
    all_accs   = [token_to_acc[t] for t in all_tokens]
    print(f"   • {len(all_tokens)} non-redundant sequences")

    # 3) Flag known enzymes (GO:0003824) via UniProt
    print("[3/5] Querying UniProt for catalytic activity (GO:0003824) …")
    enzyme_accs = uniprot_fetch_go_enzyme_flags(all_accs)
    print(f"   • {len(enzyme_accs)} sequences annotated as enzymes")

    if len(enzyme_accs) == 0:
        print("   ⚠️ No enzymes found; all sequences would be retained.")
        # Write all accessions as-is
        pd.DataFrame({"Entry": sorted(set(all_accs))}).to_csv(FINAL_CSV, index=False)
        print(f"✅ Done: {FINAL_CSV}")
        return

    # 4) Write enzyme FASTA (queries for MMseqs); use the same id tokens
    print("[4/5] Writing enzyme FASTA for MMseqs query …")
    with ENZ_FASTA.open("w") as out_fa:
        kept = 0
        for tok in all_tokens:
            if token_to_acc[tok] in enzyme_accs:
                # Use the exact token as header id (first word)
                out_fa.write(f">{tok}\n")
                seq = token_to_seq[tok]
                # wrap at 60–80 chars for readability (optional)
                for i in range(0, len(seq), 80):
                    out_fa.write(seq[i:i+80] + "\n")
                kept += 1
    print(f"   • Wrote {kept} enzyme sequences to {ENZ_FASTA}")

    # 5) MMseqs2 search: enzyme queries vs all deduplicated targets
    print("[5/5] MMseqs2 search (enzymes vs all, e<=1e-10) …")
    MM_TMP.mkdir(exist_ok=True)
    # createdb
    if not Path(str(DB_ENZ) + ".index").exists():
        run(["mmseqs", "createdb", str(ENZ_FASTA), str(DB_ENZ)])
    else:
        print("   • Reusing enzyme DB")
    if not Path(str(DB_ALL) + ".index").exists():
        run(["mmseqs", "createdb", str(DEDUP_FASTA), str(DB_ALL)])
    else:
        print("   • Reusing all-seq DB")

    # search
    if not Path(str(DB_RES)).exists():
        run([
            "mmseqs", "search", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(MM_TMP),
            "-e", str(MMSEQS_EVALUE),
            "--threads", str(MMSEQS_THREADS)
        ])
    else:
        print("   • Reusing previous search result DB")

    # convert to M8/TSV with query/target + evalue
    run([
        "mmseqs", "convertalis", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(RES_M8),
        "--format-output", "query,target,evalue,pident,alnlen"
    ])

    # Parse hits → collect all target tokens to remove (plus the enzyme queries themselves)
    print("   • Parsing MMseqs hits …")
    to_remove_tokens = set()
    with RES_M8.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            q, t, *_ = line.strip().split("\t")
            to_remove_tokens.add(t)
            to_remove_tokens.add(q)  # also remove the enzyme query protein itself

    print(f"   • Removing {len(to_remove_tokens)} tokens (enzymes + homologs)")

    # Remaining entries
    remaining_accs = sorted({token_to_acc[tok]
                             for tok in all_tokens
                             if tok not in to_remove_tokens})

    # Write final CSV
    pd.DataFrame({"Entry": remaining_accs}).to_csv(FINAL_CSV, index=False)
    print(f"\n✅ Filter completed.")
    print(f"   • Input (non-redundant): {len(all_tokens)}")
    print(f"   • Enzymes flagged (GO:0003824): {len(enzyme_accs)}")
    print(f"   • Removed by MMseqs (incl. queries): {len(to_remove_tokens)}")
    print(f"   • Remaining: {len(remaining_accs)}")
    print(f"   • Output CSV: {FINAL_CSV}")

if __name__ == "__main__":
    try:
        main()
    except sp.CalledProcessError as e:
        sys.exit(f"❌ External tool failed (return code {e.returncode}). Command: {e.cmd}")
    except Exception as e:
        sys.exit(f"❌ Error: {e}")
