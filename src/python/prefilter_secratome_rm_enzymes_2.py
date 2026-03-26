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
#INPUT_FASTA = Path("../../data/novo_smORFpipe_full_data.fasta")
#INPUT_FASTA = Path("../../data/fig_6_zebrafish_secretome/secretome_zebrafish.fasta")
INPUT_FASTA = Path("../../data/amphibian_analysis/combined_Anura_secretomes_no_ERR_DRR.fasta")
#OUT_DIR     = Path("../../data/novo_alldata_filter_out_enzymes")
#OUT_DIR     = Path("../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5")
OUT_DIR     = Path("../../data/amphibian_analysis/amphibians_filter_out_enzymes_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


tax_id = "8292"

ALL_FASTA   = INPUT_FASTA
ENZ_FASTA   = OUT_DIR / "enzymes_flagged.fasta"    # downloaded "query" set (enzymes+other exclusions)
MM_TMP      = OUT_DIR / "mmseqs_tmp"
DB_ENZ      = OUT_DIR / "enz_db"
DB_ALL      = OUT_DIR / "all_db"
DB_RES      = OUT_DIR / "enz_vs_all"
RES_M8      = OUT_DIR / "enz_vs_all.m8"
#FINAL_CSV   = OUT_DIR / "secretome_filtered_nonenzymes_nocsdhit.csv"
#FINAL_FASTA = OUT_DIR / "secretome_no_enzymes.fasta"
FINAL_CSV   = OUT_DIR / "amphibians_filtered_nonenzymes_nocsdhit.csv"
FINAL_FASTA = OUT_DIR / "amphibians_no_enzymes.fasta"

# ── CONFIG (UniProt) ───────────────────────────────────────────────────────
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# ── GO PANELS (grouped labels) ─────────────────────────────────────────────
GO_GROUPS: dict[str, list[str]] = {
    "Broad catalytic activity": ["GO:0003824"],
    "EC-class roots": ["GO:0016491","GO:0016740","GO:0016787","GO:0016829","GO:0016853","GO:0016874"],
    "Catalytic activity acting on a protein": ["GO:0140096"],
    "Proteases & families": ["GO:0008233","GO:0004175","GO:0004222","GO:0004252","GO:0004197","GO:0004190","GO:0004298"],
    "Kinases / phosphatases": ["GO:0016301","GO:0016791"],
    "Glyco & lipid hydrolases/transferases": ["GO:0004553","GO:0016757","GO:0016298"],
    "Nucleases": ["GO:0004518","GO:0004519","GO:0004527","GO:0004540","GO:0004536"],
    "Energy-hydrolyzing catalysts": ["GO:0016887","GO:0003924"],
    "Polymerases": ["GO:0003887","GO:0003899"],
    "ECM / structural": ["GO:0005201","GO:0030198","GO:0005581","GO:0005583","GO:0031012"],
    "Immune / cytokines": ["GO:0005125","GO:0008009","GO:0006956","GO:0006954"],
    "Coagulation & cascades": ["GO:0007596","GO:0072376"],
    "Protease inhibitors": ["GO:0030414","GO:0004866","GO:0004867","GO:0004869"],
    "Carriers / binders": ["GO:0008289","GO:0005319"],
    "Membrane-anchored": ["GO:0031225","GO:0016021"],
}
GO_TERMS = [t for terms in GO_GROUPS.values() for t in terms]

# ── PROTECT & DROP PANELS (on your input entries) ─────────────────────────
KEEP_GO_TERMS = [
    "GO:0005179",  # hormone activity
    "GO:0005184",  # neuropeptide hormone activity
]
TM_KEYWORDS = ["KW-0812"]                  # Transmembrane
TM_GO_TERMS = ["GO:0016021","GO:0031224"]  # integral / intrinsic membrane

# ── MMseqs2 filtering thresholds (we'll enforce these in Python) ──────────
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

def write_fasta(path: Path, token_to_seq: dict, tokens: list[str]):
    with path.open("w") as out_fa:
        for tok in tokens:
            seq = token_to_seq[tok]
            out_fa.write(f">{tok}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

def build_go_or_query(go_terms: list[str]) -> str:
    bits = [f"go:{t.split(':')[1] if ':' in t else t}" for t in go_terms]
    return "(" + " OR ".join(bits) + ")"

def build_kw_or_query(kws: list[str]) -> str:
    bits = [f"keyword:{k}" for k in kws]
    return "(" + " OR ".join(bits) + ")"

def uniprot_fetch_matching_accessions(accessions_base: list[str],
                                      go_terms: list[str] | None = None,
                                      keywords: list[str] | None = None,
                                      taxonomy_id: str | None = None,
                                      reviewed_only: bool = False) -> set[str]:
    """
    Query UniProt by GO/KW (+taxonomy) and then intersect with your input base accessions.
    (Avoids giant accession lists in the URL.)
    """
    headers = {
        "Accept": "text/tab-separated-values",
        "User-Agent": "secretome-filter/1.8 (+you@domain)"
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

def download_query_fasta(out_path: Path,
                         taxonomy_id: str | None,
                         reviewed_only: bool = False,
                         include_isoforms: bool = False,
                         max_seqs: int | None = None):
    """Download UniProt sequences matched by the OR of all GO terms across all groups."""
    scope = "ALL taxa" if taxonomy_id is None else f"tax:{taxonomy_id}"
    print(f"[2/6] Downloading UniProt query FASTA for {scope} (multi-GO query) …")

    parts = [build_go_or_query(GO_TERMS)]
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
        "User-Agent": "secretome-filter/1.7 (+you@domain)"
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
        sys.exit("❌ UniProt FASTA appears empty (no headers found).")

    size_mb = n_bytes / 1e6
    print(f"   • Wrote ~{size_mb:.1f} MB of query sequences ({scope}, {n_headers} entries) → {out_path}")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Filter a secretome FASTA; protect hormones/neuropeptides; drop TM; MMseqs homology vs multi-GO query."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download query set and overwrite existing MMseqs DBs/results.")
    parser.add_argument("--global-enzymes", action="store_true",
                        help="Use ALL taxa for the GO query set (no taxonomy filter).")
    parser.add_argument("--taxonomy-id", default=tax_id,
                        help="Taxonomy ID to restrict UniProt queries (e.g. 7955). Omit for ALL taxa.")

    parser.add_argument("--reviewed-only", action="store_true",
                        help="Download only Swiss-Prot reviewed entries (smaller, higher quality).")
    parser.add_argument("--include-isoforms", action="store_true",
                        help="Include isoform sequences in the UniProt download (bigger).")
    parser.add_argument("--max-enzyme-seqs", type=int, default=None,
                        help="Cap number of UniProt sequences (for testing / resource control).")
    args = parser.parse_args()

    if args.global_enzymes:
        args.taxonomy_id = None

    require_tool("mmseqs")
    if not ALL_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {ALL_FASTA}")

    # 1) Parse input FASTA
    print("[1/6] Parsing input FASTA and indexing sequences …")
    token_to_seq, token_to_hdr, token_to_acc, token_to_base = {}, {}, {}, {}
    for token, hdr, seq in fasta_iter_ids(ALL_FASTA):
        token_to_seq[token]  = seq
        token_to_hdr[token]  = hdr
        acc = token_to_entry(token)
        token_to_acc[token]  = acc
        token_to_base[token] = base_acc(acc)
    all_tokens = list(token_to_seq.keys())
    all_bases  = [token_to_base[t] for t in all_tokens]
    print(f"   • {len(all_tokens)} sequences total")

    secretome_tokens = set(all_tokens)

    # 2) Download UniProt query FASTA (ALL taxa or restricted)
    if args.force or not ENZ_FASTA.exists():
        tax = None if args.global_enzymes else (args.taxonomy_id or tax_id)
        if args.global_enzymes and not args.reviewed_only and args.max_enzyme_seqs is None:
            print("⚠️  Requesting ALL taxa, unreviewed+reviewed, unlimited count can be VERY large.")
            print("    Tip: add --reviewed-only and/or --max-enzyme-seqs 500000 on your first run.")
        download_query_fasta(
            ENZ_FASTA,
            taxonomy_id=tax,
            reviewed_only=args.reviewed_only,
            include_isoforms=args.include_isoforms,
            max_seqs=args.max_enzyme_seqs
        )
    else:
        print(f"[2/6] Query FASTA exists → {ENZ_FASTA} (use --force to refresh)")

    # 3) Build KEEP and TM removal sets from UniProt annotations (on your inputs)
    print("[3/6] Pulling UniProt annotations for KEEP and TM filters …")
    tax_for_targets = args.taxonomy_id  # None means ALL taxa  
    keep_bases = uniprot_fetch_matching_accessions(
        all_bases, go_terms=KEEP_GO_TERMS, keywords=None,
        taxonomy_id=tax_for_targets, reviewed_only=False
    )
    tm_bases = uniprot_fetch_matching_accessions(
        all_bases, go_terms=TM_GO_TERMS, keywords=TM_KEYWORDS,
        taxonomy_id=tax_for_targets, reviewed_only=False
    )
    keep_tokens = {t for t in all_tokens if token_to_base[t] in keep_bases}
    tm_tokens   = {t for t in all_tokens if token_to_base[t] in tm_bases}
    print(f"   • KEEP (hormone/neuropeptide) entries: {len(keep_tokens)}")
    print(f"   • Transmembrane to drop by annotation: {len(tm_tokens)}")

    # 4) MMseqs2 search: query set vs secretome
    print("[4/6] MMseqs2 search (query set vs secretome) …")
    MM_TMP.mkdir(exist_ok=True)

    if args.force:
        for p in (DB_ENZ, DB_ALL, DB_RES):
            mmseqs_db_delete(p)
        if RES_M8.exists():
            RES_M8.unlink()

    if not mmseqs_db_exists(DB_ENZ):
        run(["mmseqs", "createdb", str(ENZ_FASTA), str(DB_ENZ)])
    else:
        print("   • Query DB exists")

    if not mmseqs_db_exists(DB_ALL):
        run(["mmseqs", "createdb", str(ALL_FASTA), str(DB_ALL)])
    else:
        print("   • Secretome DB exists")

    if not mmseqs_db_exists(DB_RES):
        run([
            "mmseqs", "search", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(MM_TMP),
            "-e", str(MMSEQS_EVALUE),
            "--threads", str(MMSEQS_THREADS)
        ])
    else:
        print("   • Result DB exists; skipping search")

    # Export ALL alignments; we'll filter in Python
    run([
    "mmseqs", "convertalis", str(DB_ENZ), str(DB_ALL), str(DB_RES), str(RES_M8),
    "--format-output", "query,target,evalue,pident,alnlen,qcov,tcov,qlen,tlen"
    ])

    # 5) Parse hits → apply ID/COV/LEN thresholds; remove targets (+ overlapping queries) but NEVER KEEP
    print("[5/6] Parsing MMseqs hits …")
    to_remove_tokens = set(tm_tokens)  # start with TM-annotated
    n_tm_removed     = len(tm_tokens)
    n_target_hits = 0
    n_query_overlap = 0

    pid_thresh = MIN_SEQ_ID * 100.0  # pident is in percent
    with RES_M8.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            q, t, e, pid, alnlen, qcov, tcov, qlen, tlen = line.rstrip("\n").split("\t")
            # normalize numeric fields
            try:
                pid   = float(pid)
                alnlen= int(float(alnlen))
                qcov  = float(qcov);  qcov = qcov/100.0 if qcov > 1 else qcov
                tcov  = float(tcov);  tcov = tcov/100.0 if tcov > 1 else tcov
            except ValueError:
                continue

            # apply thresholds
            if alnlen < MIN_ALN_LEN:            # too short
                continue
            if pid < pid_thresh:                # identity too low
                continue
            if tcov < MIN_COVERAGE:             # target coverage too low
                continue

            qtok = q
            ttok = t

            # remove target unless protected
            if ttok not in keep_tokens and ttok not in to_remove_tokens:
                to_remove_tokens.add(ttok)
                n_target_hits += 1

            # ALSO remove query token if it's literally present in your secretome (unless protected)
            if qtok in secretome_tokens and qtok not in keep_tokens and qtok not in to_remove_tokens:
                to_remove_tokens.add(qtok)
                n_query_overlap += 1

    # 6) Write outputs
    print("[6/6] Writing outputs …")
    remaining_tokens = [t for t in all_tokens if t not in to_remove_tokens]
    remaining_accs   = [token_to_acc[t] for t in remaining_tokens]

    pd.DataFrame({"Entry": sorted(set(remaining_accs))}).to_csv(FINAL_CSV, index=False)

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
    print(f"   • Removed (TM annotation): {n_tm_removed}")
    print(f"   • Removed by MMseqs (targets): {n_target_hits}")
    print(f"   • Removed by MMseqs (overlapping queries): {n_query_overlap}")
    print(f"   • Protected (KEEP) entries: {len(keep_tokens)}")
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
