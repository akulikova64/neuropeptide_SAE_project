#!/usr/bin/env python3
import os
import sys
import re
import time
import shutil
import argparse
import subprocess as sp
from pathlib import Path

import requests
import pandas as pd

# ── CONFIG (paths) ─────────────────────────────────────────────────────────
INPUT_FASTA = Path("../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta")
OUT_DIR     = Path("../../data/secretome_filter_out_nocdhit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FASTA  = INPUT_FASTA                         # use input directly (no CD-HIT)
ENZ_FASTA  = OUT_DIR / "enzymes_flagged.fasta"
MM_TMP     = OUT_DIR / "mmseqs_tmp"
DB_ENZ     = OUT_DIR / "enz_db"
DB_ALL     = OUT_DIR / "all_db"
DB_RES     = OUT_DIR / "enz_vs_all"
RES_M8     = OUT_DIR / "enz_vs_all.m8"
FINAL_CSV  = OUT_DIR / "secretome_filtered_nonenzymes_nocsdhit.csv"

# ── CONFIG (UniProt) ───────────────────────────────────────────────────────
TAXON_ID       = "9606"
GO_TERM        = "GO:0003824"   # catalytic activity
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# ── CONFIG (MMseqs2) ───────────────────────────────────────────────────────
MMSEQS_EVALUE  = 1e-10
MMSEQS_THREADS = max(1, os.cpu_count() or 1)

# ── UTILITIES ───────────────────────────────────────────────────────────────
def require_tool(name: str):
    if shutil.which(name) is None:
        sys.exit(f"❌ Required tool '{name}' not found in PATH.")

def fasta_iter_ids(path: Path):
    """Yield (id_token, header_line, sequence) where id_token is first word after '>'."""
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
    UniProt FASTA style: >sp|P12345|NAME  or >tr|A0A...|NAME  or >P12345-2 ...
    Return accession between '|' when present; else fall back to id_token.
    """
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token

def base_acc(acc: str) -> str:
    """Return base UniProt accession (strip isoform suffix like '-2')."""
    return re.sub(r"-\d+$", "", acc)

def run(cmd, **kwargs):
    print("▶", " ".join(map(str, cmd)))
    sp.run(cmd, check=True, **kwargs)

def mmseqs_db_exists(prefix: Path) -> bool:
    """Return True if MMseqs DB with given prefix exists (either .dbtype or .index)."""
    return Path(f"{prefix}.dbtype").exists() or Path(f"{prefix}.index").exists()

def mmseqs_db_delete(prefix: Path):
    """Delete common MMseqs DB artifacts for the given prefix."""
    for ext in (".dbtype", ".index", ".lookup", ".source", ".h", ".ca", ".seq", ".aln"):
        p = Path(f"{prefix}{ext}")
        if p.exists():
            p.unlink()

def uniprot_fetch_go_enzyme_flags_base(accessions, include_kw_secreted: bool = True):
    """
    Return *base* accessions annotated with catalytic activity (GO:0003824)
    among the provided accessions (which may include isoforms like 'P12345-2').
    """
    headers = {
        "Accept": "text/tab-separated-values",
        "User-Agent": "secretome-filter/1.2 (+you@domain)"
    }
    # Build UniProt query
    parts = [f"(taxonomy_id:{TAXON_ID})", f"(go:{GO_TERM.split(':')[1]})"]
    if include_kw_secreted:
        parts.insert(1, "(keyword:KW-0964)")
    query = " AND ".join(parts)

    params = {"query": query, "fields": "accession", "format": "tsv"}

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
        r.raise_for_status()

    enzyme_all_base = set()
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
            enzyme_all_base.add(base_acc(parts[acc_idx]))

    acc_bases = {base_acc(a) for a in accessions}
    return enzyme_all_base & acc_bases

def write_fasta(path: Path, token_to_seq: dict, tokens: list[str]):
    with path.open("w") as out_fa:
        for tok in tokens:
            seq = token_to_seq[tok]
            out_fa.write(f">{tok}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Filter secretome by removing enzymes and homologs (no CD-HIT).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite any existing MMseqs DBs/results and rerun search.")
    parser.add_argument("--no-kw-secreted", action="store_true",
                        help="Do NOT include the UniProt Secreted keyword (KW-0964) in the query.")
    args = parser.parse_args()

    require_tool("mmseqs")
    if not ALL_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {ALL_FASTA}")

    # 1) Parse input FASTA
    print("[1/4] Parsing input FASTA and collecting accessions …")
    token_to_seq, token_to_hdr, token_to_acc, token_to_base = {}, {}, {}, {}
    for token, hdr, seq in fasta_iter_ids(ALL_FASTA):
        token_to_seq[token]  = seq
        token_to_hdr[token]  = hdr
        acc = token_to_entry(token)         # may be isoform, e.g., 'P12345-2'
        token_to_acc[token]  = acc
        token_to_base[token] = base_acc(acc)
    all_tokens = list(token_to_seq.keys())
    all_accs   = [token_to_acc[t] for t in all_tokens]
    print(f"   • {len(all_tokens)} sequences total")

    # 2) UniProt enzyme detection (by BASE accession)
    print("[2/4] Querying UniProt for catalytic activity (GO:0003824) …")
    enzyme_base_accs = uniprot_fetch_go_enzyme_flags_base(
        all_accs, include_kw_secreted=not args.no_kw_secreted
    )
    print(f"   • {len(enzyme_base_accs)} sequences annotated as enzymes (base accessions)")
    if enzyme_base_accs:
        ex = ", ".join(sorted(list(enzyme_base_accs))[:10])
        print(f"     e.g., {ex}")

    if len(enzyme_base_accs) == 0:
        print("   ⚠️ No enzymes found; all sequences would be retained.")
        pd.DataFrame({"Entry": sorted(set(all_accs))}).to_csv(FINAL_CSV, index=False)
        print(f"✅ Done: {FINAL_CSV}")
        return

    # 3) Write enzyme FASTA for MMseqs queries (by BASE accession)
    print("[3/4] Writing enzyme FASTA for MMseqs query …")
    enz_tokens = [tok for tok in all_tokens if token_to_base[tok] in enzyme_base_accs]
    write_fasta(ENZ_FASTA, token_to_seq, enz_tokens)
    print(f"   • Wrote {len(enz_tokens)} enzyme sequences → {ENZ_FASTA}")
    if not enz_tokens:
        print("   ⚠️ None of your FASTA tokens matched enzyme base accessions. "
              "Check header format or try --no-kw-secreted.")

    # 4) MMseqs2 search: enzymes vs all
    print("[4/4] MMseqs2 search (enzymes vs all, e<=1e-10) …")
    MM_TMP.mkdir(exist_ok=True)

    # createdb: enzyme DB
    if args.force:
        mmseqs_db_delete(DB_ENZ)
    if not mmseqs_db_exists(DB_ENZ):
        run(["mmseqs", "createdb", str(ENZ_FASTA), str(DB_ENZ)])
    else:
        print("   • Enzyme DB exists")

    # createdb: all DB
    if args.force:
        mmseqs_db_delete(DB_ALL)
    if not mmseqs_db_exists(DB_ALL):
        run(["mmseqs", "createdb", str(ALL_FASTA), str(DB_ALL)])
    else:
        print("   • All-seq DB exists")

    # search result DB
    if args.force:
        mmseqs_db_delete(DB_RES)
        if RES_M8.exists():
            RES_M8.unlink()

    if not mmseqs_db_exists(DB_RES):
        run([
            "mmseqs", "search", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(MM_TMP),
            "-e", str(MMSEQS_EVALUE),
            "--threads", str(MMSEQS_THREADS)
        ])
    else:
        print("   • Result DB exists; skipping search (use --force to overwrite)")

    # Always regenerate tabular hits with original FASTA headers.
    # CRITICAL: use qheader/theader so we can match tokens from FASTA.
    run([
        "mmseqs", "convertalis", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(RES_M8),
        "--format-output", "qheader,theader,evalue,pident,alnlen"
    ])

    # Parse hits → collect tokens (enzyme queries + homolog targets) to remove
    print("   • Parsing MMseqs hits …")
    to_remove_tokens = set()
    with RES_M8.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            qh, th, *_ = line.rstrip("\n").split("\t")
            qtok = qh.split()[0]  # first token of header (matches FASTA token_to_* keys)
            ttok = th.split()[0]
            to_remove_tokens.add(qtok)
            to_remove_tokens.add(ttok)

    print(f"   • Removing {len(to_remove_tokens)} tokens (enzymes + homologs)")

    # Remaining entries: report original accessions from FASTA
    remaining_accs = sorted({
        token_to_acc[tok] for tok in all_tokens if tok not in to_remove_tokens
    })

    # Write final CSV
    pd.DataFrame({"Entry": remaining_accs}).to_csv(FINAL_CSV, index=False)

    # Summary
    print(f"\n✅ Filter completed.")
    print(f"   • Input sequences: {len(all_tokens)}")
    print(f"   • Enzymes flagged (base acc): {len(enzyme_base_accs)}")
    print(f"   • Enzyme tokens written: {len(enz_tokens)}")
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
