#!/usr/bin/env python3
"""
Filter immunoglobulins (and homologs) out of the secretome non-enzyme set.

Assumes the previous enzyme-filtering script has already been run and that
these files exist:

    ../../data/secretome_filter_out_enzymes_5/secretome_filtered_nonenzymes_nocsdhit.csv
    ../../data/secretome_filter_out_enzymes_5/secretome_no_enzymes.fasta

Outputs:

    ../../data/secretome_filter_out_enzymes_5/secretome_no_immonoglobulins.csv
"""

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

# ── here are my paths ─────────────────────────────────────────────────────────

#OUT_DIR = Path("../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5")
OUT_DIR     = Path("../../data/amphibian_analysis/amphibians_filter_out_enzymes_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tax_id = "8292"

#INPUT_CSV   = OUT_DIR / "secretome_filtered_nonenzymes_nocsdhit.csv"
#INPUT_FASTA = OUT_DIR / "secretome_no_enzymes.fasta"
INPUT_CSV   = OUT_DIR / "amphibians_filtered_nonenzymes_nocsdhit.csv"
INPUT_FASTA = OUT_DIR / "amphibians_no_enzymes.fasta"

IG_FASTA   = OUT_DIR / "immunoglobulin_query.fasta"
MM_TMP     = OUT_DIR / "mmseqs_tmp_ig"
DB_IG      = OUT_DIR / "ig_db"
DB_ALL     = OUT_DIR / "secretome_db_ig"
DB_RES     = OUT_DIR / "ig_vs_secretome"
RES_M8     = OUT_DIR / "ig_vs_secretome.m8"
#FINAL_CSV  = OUT_DIR / "secretome_no_immonoglobulins.csv"
FINAL_CSV  = OUT_DIR / "amphibians_no_immonoglobulins.csv"
FINAL_FASTA = OUT_DIR / "amphibians_no_immonoglobulins.fasta"

# ── CONFIG (UniProt) ───────────────────────────────────────────────────────

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# Immunoglobulin-related annotation selectors
IG_GO_TERMS = [
    "GO:0019814",  # immunoglobulin complex
]
IG_KEYWORDS = [
    "KW-0377",     # Immunoglobulin domain
]

# ── MMseqs2 filtering parameters (same as in my enzyme script) ───────────────────

MMSEQS_EVALUE   = 1e-10
MMSEQS_THREADS  = max(1, os.cpu_count() or 1)
MIN_SEQ_ID      = 0.35     # fraction; e.g., 0.35 = 35% identity
MIN_COVERAGE    = 0.60     # fraction of TARGET covered by alignment
MIN_ALN_LEN     = 20       # minimum aligned length (aa) to accept a hit

# ── UTILITIES ───────────────────────────────────────────────────────────────

def require_tool(name: str):
    if shutil.which(name) is None:
        sys.exit(f"❌ Required tool '{name}' not found in PATH.")

def fasta_iter_ids(path: Path):
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
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token

def base_acc(acc: str) -> str:
    return re.sub(r"-\d+$", "", acc)

def run(cmd, **kwargs):
    print("▶", " ".join(map(str, cmd)))
    sp.run(cmd, check=True, **kwargs)

def mmseqs_db_exists(prefix: Path) -> bool:
    return Path(f"{prefix}.dbtype").exists() or Path(f"{prefix}.index").exists()

def mmseqs_db_delete(prefix: Path):
    for ext in (".dbtype", ".index", ".lookup", ".source", ".h", ".ca", ".seq", ".aln"):
        p = Path(f"{prefix}{ext}")
        if p.exists():
            p.unlink()

def build_go_or_query(go_terms: list[str]) -> str:
    bits = [f"go:{t.split(':')[1] if ':' in t else t}" for t in go_terms]
    return "(" + " OR ".join(bits) + ")"

def build_kw_or_query(kws: list[str]) -> str:
    bits = [f"keyword:{k}" for k in kws]
    return "(" + " OR ".join(bits) + ")"

def uniprot_fetch_matching_accessions(accessions_base: list[str],
                                      go_terms: list[str] | None = None,
                                      keywords: list[str] | None = None,
                                      taxonomy_id: str | None = tax_id,
                                      reviewed_only: bool = False) -> set[str]:
    """
    Query UniProt by GO/KW (+taxonomy) and intersect with your input base accessions.
    """
    headers = {
        "Accept": "text/tab-separated-values",
        "User-Agent": "secretome-ig-filter/1.0 (+you@domain)"
    }
    if not accessions_base:
        return set()

    selectors = []
    if go_terms:
        selectors.append(build_go_or_query(go_terms))
    if keywords:
        selectors.append(build_kw_or_query(keywords))
    if not selectors:
        return set()

    parts = ["(" + " OR ".join(selectors) + ")"]
    if taxonomy_id:
        parts.insert(0, f"(taxonomy_id:{taxonomy_id})")
    if reviewed_only:
        parts.append("(reviewed:true)")
    query = " AND ".join(parts)

    params = {"query": query, "fields": "accession", "format": "tsv"}
    for attempt in range(3):
        try:
            r = requests.get(UNIPROT_STREAM, params=params, headers=headers,
                             timeout=180, stream=True)
            if r.status_code == 200:
                break
            time.sleep(1 + attempt)
        except requests.RequestException:
            time.sleep(1 + attempt)
    else:
        r.raise_for_status()

    hits = set()
    first = True
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if first:
            first = False
            continue
        acc = line.split("\t", 1)[0].strip()
        hits.add(base_acc(acc))

    input_bases = set(accessions_base)
    return hits & input_bases

def download_ig_query_fasta(out_path: Path,
                            taxonomy_id: str | None,
                            reviewed_only: bool = False,
                            include_isoforms: bool = False,
                            max_seqs: int | None = None):
    """
    Download UniProt sequences that are annotated as immunoglobulins or
    immunoglobulin-domain containing.
    """
    scope = "ALL taxa" if taxonomy_id is None else f"tax:{taxonomy_id}"
    print(f"[2/5] Downloading UniProt immunoglobulin query FASTA for {scope} …")

    selectors = []
    if IG_GO_TERMS:
        selectors.append(build_go_or_query(IG_GO_TERMS))
    if IG_KEYWORDS:
        selectors.append(build_kw_or_query(IG_KEYWORDS))

    if not selectors:
        sys.exit("❌ No immunoglobulin selectors configured.")

    parts = ["(" + " OR ".join(selectors) + ")"]
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
        "User-Agent": "secretome-ig-filter/1.0 (+you@domain)"
    }

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
        sys.exit("❌ UniProt immunoglobulin FASTA appears empty (no headers found).")

    size_mb = n_bytes / 1e6
    print(f"   • Wrote ~{size_mb:.1f} MB of immunoglobulin sequences ({scope}, {n_headers} entries) → {out_path}")

# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter immunoglobulins (and homologs) from secretome non-enzyme set."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download immunoglobulin query set and overwrite existing MMseqs DBs/results.")
    parser.add_argument("--global-igs", action="store_true",
                        help="Use ALL taxa for the immunoglobulin query set (no taxonomy filter).")
    parser.add_argument("--taxonomy-id", default=tax_id,
                        help="Taxonomy to restrict immunoglobulin query to (ignored if --global-igs). Default: tax_id.")
    parser.add_argument("--reviewed-only", action="store_true",
                        help="Download only Swiss-Prot reviewed immunoglobulin entries.")
    parser.add_argument("--include-isoforms", action="store_true",
                        help="Include isoform sequences in the UniProt immunoglobulin download.")
    parser.add_argument("--max-ig-seqs", type=int, default=None,
                        help="Cap number of UniProt immunoglobulin sequences (for testing / resource control).")
    args = parser.parse_args()

    require_tool("mmseqs")

    if not INPUT_CSV.exists():
        sys.exit(f"❌ Input CSV not found: {INPUT_CSV}")
    if not INPUT_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {INPUT_FASTA}")

    # 1) Load CSV & FASTA
    print("[1/5] Loading secretome entries and sequences …")
    df = pd.read_csv(INPUT_CSV)
    if "Entry" not in df.columns:
        sys.exit("❌ CSV does not contain an 'Entry' column.")

    csv_entries = set(df["Entry"].astype(str))
    print(f"   • Entries from CSV: {len(csv_entries)}")

    token_to_seq: dict[str, str] = {}
    token_to_hdr: dict[str, str] = {}
    token_to_acc: dict[str, str] = {}
    token_to_base: dict[str, str] = {}

    for token, hdr, seq in fasta_iter_ids(INPUT_FASTA):
        acc = token_to_entry(token)
        base = base_acc(acc)
        token_to_seq[token] = seq
        token_to_hdr[token] = hdr
        token_to_acc[token] = acc
        token_to_base[token] = base

    all_tokens = list(token_to_seq.keys())
    all_bases  = [token_to_base[t] for t in all_tokens]
    secretome_tokens = set(all_tokens)

    print(f"   • Sequences in FASTA: {len(all_tokens)}")

    # 2) Download immunoglobulin query FASTA (ALL taxa or restricted)
    if args.force or not IG_FASTA.exists():
        tax = None if args.global_igs else (args.taxonomy_id or "9606")
        if args.global_igs and not args.reviewed_only and args.max_ig_seqs is None:
            print("⚠️  Requesting ALL taxa, unreviewed+reviewed, unlimited count can be VERY large.")
            print("    Tip: add --reviewed-only and/or --max-ig-seqs 500000 on your first run.")
        download_ig_query_fasta(
            IG_FASTA,
            taxonomy_id=tax,
            reviewed_only=args.reviewed_only,
            include_isoforms=args.include_isoforms,
            max_seqs=args.max_ig_seqs,
        )
    else:
        print(f"[2/5] Immunoglobulin query FASTA exists → {IG_FASTA} (use --force to refresh)")

    # 3) Pull UniProt annotations to flag directly annotated immunoglobulins in your set
    print("[3/5] Pulling UniProt annotations for immunoglobulin filters …")
    tax_for_targets = args.taxonomy_id  # already defaults to 8292
    ig_bases = uniprot_fetch_matching_accessions(
        all_bases, go_terms=IG_GO_TERMS, keywords=IG_KEYWORDS,
        taxonomy_id=tax_for_targets, reviewed_only=False
    )
    ig_tokens_annot = {t for t in all_tokens if token_to_base[t] in ig_bases}
    print(f"   • Directly annotated immunoglobulin entries in secretome: {len(ig_tokens_annot)}")

    # 4) MMseqs2 search: immunoglobulin set vs secretome
    print("[4/5] MMseqs2 search (immunoglobulin set vs secretome) …")
    MM_TMP.mkdir(exist_ok=True)

    if args.force:
        for p in (DB_IG, DB_ALL, DB_RES):
            mmseqs_db_delete(p)
        if RES_M8.exists():
            RES_M8.unlink()

    if not mmseqs_db_exists(DB_IG):
        run(["mmseqs", "createdb", str(IG_FASTA), str(DB_IG)])
    else:
        print("   • Immunoglobulin query DB exists")

    if not mmseqs_db_exists(DB_ALL):
        run(["mmseqs", "createdb", str(INPUT_FASTA), str(DB_ALL)])
    else:
        print("   • Secretome DB (for Ig filtering) exists")

    if not mmseqs_db_exists(DB_RES):
        run([
            "mmseqs", "search", str(DB_IG), str(DB_ALL), str(DB_RES), str(MM_TMP),
            "-e", str(MMSEQS_EVALUE),
            "--threads", str(MMSEQS_THREADS)
        ])
    else:
        print("   • Result DB exists; skipping search")

    # Export ALL alignments; we'll filter in Python
    run([
        "mmseqs", "convertalis", str(DB_IG), str(DB_ALL), str(DB_RES), str(RES_M8),
        "--format-output", "query,target,evalue,pident,alnlen,qcov,tcov,qlen,tlen"
    ])

    # 5) Parse hits → apply ID/COV/LEN thresholds; remove targets and overlapping queries
    print("[5/5] Parsing MMseqs hits and writing outputs …")
    to_remove_tokens = set(ig_tokens_annot)  # start with directly annotated Ig
    n_annot_removed  = len(ig_tokens_annot)
    n_target_hits    = 0
    n_query_overlap  = 0

    pid_thresh = MIN_SEQ_ID * 100.0  # pident is in percent
    if RES_M8.exists():
        with RES_M8.open() as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                q, t, e, pid, alnlen, qcov, tcov, qlen, tlen = line.rstrip("\n").split("\t")
                # normalize numeric fields
                try:
                    pid    = float(pid)
                    alnlen = int(float(alnlen))
                    qcov   = float(qcov);  qcov = qcov/100.0 if qcov > 1 else qcov
                    tcov   = float(tcov);  tcov = tcov/100.0 if tcov > 1 else tcov
                except ValueError:
                    continue

                # apply thresholds
                if alnlen < MIN_ALN_LEN:
                    continue
                if pid < pid_thresh:
                    continue
                if tcov < MIN_COVERAGE:
                    continue

                qtok = q
                ttok = t

                # remove target if it's in our secretome and not already marked
                if ttok in secretome_tokens and ttok not in to_remove_tokens:
                    to_remove_tokens.add(ttok)
                    n_target_hits += 1

                # ALSO remove query token if it's literally present in your secretome
                if qtok in secretome_tokens and qtok not in to_remove_tokens:
                    to_remove_tokens.add(qtok)
                    n_query_overlap += 1

    # Build remaining set and write CSV
    remaining_tokens = [t for t in all_tokens if t not in to_remove_tokens]
    remaining_accs   = [token_to_acc[t] for t in remaining_tokens]

    # Work in terms of unique Entry IDs overlapping with original CSV
    remaining_entries = {acc for acc in remaining_accs if acc in csv_entries}
    removed_entries   = csv_entries - remaining_entries

    # Write final CSV (unique Entry IDs)
    pd.DataFrame({"Entry": sorted(remaining_entries)}).to_csv(FINAL_CSV, index=False)

    # Write filtered FASTA
    with FINAL_FASTA.open("w") as out_fa:
        for tok in remaining_tokens:
            hdr = token_to_hdr[tok]
            seq = token_to_seq[tok]
            out_fa.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

    # Summary
    print("\n✅ Immunoglobulin filter completed.")
    print(f"   • Input peptides (unique Entry from CSV): {len(csv_entries)}")
    print(f"   • Removed by UniProt Ig annotation (token-level): {n_annot_removed}")
    print(f"   • Removed by MMseqs (targets): {n_target_hits}")
    print(f"   • Removed by MMseqs (overlapping queries): {n_query_overlap}")
    print(f"   • Total peptides removed (unique Entry): {len(removed_entries)}")
    print(f"   • Remaining peptides (unique Entry): {len(remaining_entries)}")
    print(f"   • Output CSV: {FINAL_CSV}")
    print(f"   • Output FASTA: {FINAL_FASTA}")

if __name__ == "__main__":
    try:
        main()
    except sp.CalledProcessError as e:
        sys.exit(f"❌ External tool failed (return code {e.returncode}). Command: {e.cmd}")
    except Exception as e:
        sys.exit(f"❌ Error: {e}")