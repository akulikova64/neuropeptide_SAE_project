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
OUT_DIR     = Path("../../data/secretome_filter_out_enzymes_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FASTA   = INPUT_FASTA                          # use input directly (no CD-HIT)
ENZ_FASTA   = OUT_DIR / "enzymes_flagged.fasta"    # downloaded enzymes (query set)
MM_TMP      = OUT_DIR / "mmseqs_tmp"
DB_ENZ      = OUT_DIR / "enz_db"
DB_ALL      = OUT_DIR / "all_db"
DB_RES      = OUT_DIR / "enz_vs_all"
RES_M8      = OUT_DIR / "enz_vs_all.m8"
FINAL_CSV   = OUT_DIR / "secretome_filtered_nonenzymes_nocsdhit.csv"
FINAL_FASTA = OUT_DIR / "secretome_no_enzymes.fasta"

# ── CONFIG (UniProt) ───────────────────────────────────────────────────────
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

def write_fasta(path: Path, token_to_seq: dict, tokens: list[str]):
    with path.open("w") as out_fa:
        for tok in tokens:
            seq = token_to_seq[tok]
            out_fa.write(f">{tok}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

def download_enzyme_fasta(out_path: Path,
                          taxonomy_id: str | None,
                          reviewed_only: bool = False,
                          include_isoforms: bool = False,
                          max_seqs: int | None = None):
    """
    Download enzymes from UniProt as FASTA.

    Args:
      taxonomy_id: e.g., "9606" for human; None for ALL taxa.
      reviewed_only: Swiss-Prot only if True (default False).
      include_isoforms: include isoforms if True (default False).
      max_seqs: write at most this many sequences (None = all).

    Query examples:
      ALL taxa:       (go:0003824) [± reviewed:true] [± includeIsoform]
      Human only:     (taxonomy_id:9606) AND (go:0003824) [...]
    """
    if taxonomy_id is None:
        print("[2/4] Downloading UniProt enzyme FASTA for ALL taxa (GO:0003824) …")
    else:
        print(f"[2/4] Downloading UniProt enzyme FASTA for taxonomy_id:{taxonomy_id} …")

    # Build query
    parts = [f"(go:{GO_TERM.split(':')[1]})"]
    if taxonomy_id:
        parts.insert(0, f"(taxonomy_id:{taxonomy_id})")
    if reviewed_only:
        parts.append("(reviewed:true)")
    query = " AND ".join(parts)

    params = {"query": query, "format": "fasta"}
    if include_isoforms:
        params["includeIsoform"] = "true"

    headers = {
        "Accept": "text/x-fasta",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "secretome-filter/1.4 (+you@domain)"
    }

    # Stream line-wise so we can optionally cap at max_seqs
    for attempt in range(4):
        try:
            r = requests.get(UNIPROT_STREAM, params=params, headers=headers,
                             timeout=300, stream=True)
            if r.status_code == 200:
                break
            print(f"   • UniProt replied {r.status_code}; retrying …")
            time.sleep(2 * (attempt + 1))
        except requests.RequestException as e:
            print(f"   • Request error: {e}; retrying …")
            time.sleep(2 * (attempt + 1))
    else:
        r.raise_for_status()

    n_headers = 0
    n_bytes   = 0
    with out_path.open("wt", encoding="utf-8") as fh:
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith(">"):
                n_headers += 1
                if max_seqs is not None and n_headers > max_seqs:
                    break
            fh.write(line + "\n")
            n_bytes += len(line) + 1

    if n_headers == 0:
        sys.exit("❌ UniProt FASTA appears empty (no headers found).")

    size_mb = n_bytes / 1e6
    if taxonomy_id is None:
        print(f"   • Wrote ~{size_mb:.1f} MB of enzyme sequences "
              f"(ALL taxa, {n_headers} entries) → {out_path}")
    else:
        print(f"   • Wrote ~{size_mb:.1f} MB of enzyme sequences "
              f"(tax:{taxonomy_id}, {n_headers} entries) → {out_path}")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Filter a human secretome FASTA by removing anything homologous to UniProt enzymes."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download enzymes and overwrite existing MMseqs DBs/results.")
    parser.add_argument("--global-enzymes", action="store_true",
                        help="Use ALL UniProt enzymes (no taxonomy filter).")
    parser.add_argument("--taxonomy-id", default="9606",
                        help="Taxonomy to restrict enzymes to (ignored if --global-enzymes). Default: 9606 (human).")
    parser.add_argument("--reviewed-only", action="store_true",
                        help="Download only Swiss-Prot reviewed entries (smaller, higher quality).")
    parser.add_argument("--include-isoforms", action="store_true",
                        help="Include isoform sequences in the UniProt download (bigger).")
    parser.add_argument("--max-enzyme-seqs", type=int, default=None,
                        help="Cap number of enzyme sequences (for testing / resource control).")
    args = parser.parse_args()

    require_tool("mmseqs")
    if not ALL_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {ALL_FASTA}")

    # 1) Parse input FASTA
    print("[1/4] Parsing input FASTA and indexing sequences …")
    token_to_seq, token_to_hdr, token_to_acc = {}, {}, {}
    for token, hdr, seq in fasta_iter_ids(ALL_FASTA):
        token_to_seq[token] = seq
        token_to_hdr[token] = hdr
        token_to_acc[token] = token_to_entry(token)
    all_tokens = list(token_to_seq.keys())
    print(f"   • {len(all_tokens)} sequences total")

    # ★ CHANGED: cache secretome tokens for overlap checks with queries
    secretome_tokens = set(all_tokens)  # tokens exactly as they appear in your FASTA

    # 2) Download enzyme FASTA (ALL taxa or restricted)
    if args.force or not ENZ_FASTA.exists():
        tax = None if args.global_enzymes else (args.taxonomy_id or "9606")
        if args.global_enzymes and not args.reviewed_only and args.max_enzyme_seqs is None:
            print("⚠️  Requesting ALL taxa, unreviewed+reviewed, unlimited count can be VERY large.")
            print("    Tip: add --reviewed-only and/or --max-enzyme-seqs 500000 on your first run.")
        download_enzyme_fasta(
            ENZ_FASTA,
            taxonomy_id=tax,
            reviewed_only=args.reviewed_only,
            include_isoforms=args.include_isoforms,
            max_seqs=args.max_enzyme_seqs
        )
    else:
        print(f"[2/4] Enzyme FASTA exists → {ENZ_FASTA} (use --force to refresh)")

    # 3) MMseqs2 search: enzymes vs all
    print("[3/4] MMseqs2 search (enzymes vs your secretome, e<=1e-10) …")
    MM_TMP.mkdir(exist_ok=True)

    # createdb: enzyme DB
    if args.force:
        mmseqs_db_delete(DB_ENZ)
    if not mmseqs_db_exists(DB_ENZ):
        run(["mmseqs", "createdb", str(ENZ_FASTA), str(DB_ENZ)])
    else:
        print("   • Enzyme DB exists")

    # createdb: all DB (your secretome)
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
    run([
        "mmseqs", "convertalis", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(RES_M8),
        "--format-output", "qheader,theader,evalue,pident,alnlen"
    ])

    # ★ CHANGED: Parse hits → collect TARGET tokens and overlapping QUERY tokens to remove
    print("[4/4] Parsing MMseqs hits and writing outputs …")
    to_remove_tokens = set()
    n_target_hits = 0
    n_query_overlap = 0
    with RES_M8.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qh = parts[0]                     # qheader = enzyme
            th = parts[1]                     # theader = your secretome entry
            qtok = qh.split()[0]
            ttok = th.split()[0]
            # always remove the target token (your secretome)
            if ttok not in to_remove_tokens:
                to_remove_tokens.add(ttok)
                n_target_hits += 1
            # ALSO remove the query token if its header token appears in your secretome FASTA
            if qtok in secretome_tokens and qtok not in to_remove_tokens:
                to_remove_tokens.add(qtok)
                n_query_overlap += 1

    print(f"   • Removing {len(to_remove_tokens)} tokens "
          f"(targets: {n_target_hits}, overlapping queries: {n_query_overlap})")

    # Remaining entries
    remaining_tokens = [t for t in all_tokens if t not in to_remove_tokens]
    remaining_accs   = [token_to_acc[t] for t in remaining_tokens]

    # Write CSV
    pd.DataFrame({"Entry": sorted(set(remaining_accs))}).to_csv(FINAL_CSV, index=False)

    # Write FASTA
    with FINAL_FASTA.open("w") as out_fa:
        for tok in remaining_tokens:
            hdr = token_to_hdr[tok]
            seq = token_to_seq[tok]
            out_fa.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

    # Summary
    print(f"\n✅ Filter completed.")
    print(f"   • Input sequences: {len(all_tokens)}")
    print(f"   • Removed by MMseqs (targets + overlapping queries): {len(to_remove_tokens)}")
    print(f"   • Remaining: {len(remaining_tokens)}")
    print(f"   • Output CSV:   {FINAL_CSV}")
    print(f"   • Output FASTA: {FINAL_FASTA}")

if __name__ == "__main__":
    try:
        main()
    except sp.CalledProcessError as e:
        sys.exit(f"❌ External tool failed (return code {e.returncode}). Command: {e.cmd}")
    except Exception as e:
        sys.exit(f"❌ Error: {e}")
