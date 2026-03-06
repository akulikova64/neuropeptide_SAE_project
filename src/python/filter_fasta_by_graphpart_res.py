#!/usr/bin/env python3
"""
Make a 1:1 FASTA matching GraphPart ACs when original FASTA has headers like:
  >52904_0:00337b {...}
  >52904_0:002ff4 {...}
  >52904_0 {...}

Policy (per AC):
  1) If an exact header '>AC' exists, use that.
  2) Else, among headers starting with '>AC:', choose the LONGEST sequence.

Outputs exactly one sequence per AC in the CSV (if found).
"""

import argparse
import csv
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def open_auto(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def parse_base_id(header_line: str) -> str:
    """
    Get base ID from a FASTA header:
      - take first token after '>' up to whitespace
      - trim at first ':' if present
    Examples:
      >52904_0:00337b {...} → 52904_0
      >52904_0 {...}        → 52904_0
    """
    tok = header_line[1:].strip().split()[0]
    if ":" in tok:
        tok = tok.split(":", 1)[0]
    return tok

def iter_fasta(path: str):
    with open_auto(path, "rt") as fh:
        header = None
        chunks = []
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line.strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)

def load_acs(csv_path: str) -> Set[str]:
    keep: Set[str] = set()
    with open(csv_path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = False
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            pass
        if has_header:
            reader = csv.DictReader(f)
            ac_col = None
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name and name.strip().lower() == "ac":
                        ac_col = name
                        break
                if ac_col is None:
                    ac_col = reader.fieldnames[0]
            for row in reader:
                val = (row.get(ac_col, "") if ac_col else "").strip()
                if val:
                    keep.add(val)
        else:
            f.seek(0)
            for row in csv.reader(f):
                if row and row[0].strip():
                    keep.add(row[0].strip())
    return keep

def write_fasta(records: List[Tuple[str,str]], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w") as out:
        for hdr, seq in records:
            out.write(hdr + "\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")
            n += 1
    return n

def choose_representatives(
    acs: Set[str],
    fasta_path: str,
    policy: str = "longest"  # alternatives: "first", "lex"
) -> Tuple[List[Tuple[str,str]], dict]:
    """
    Build a list of (header, seq), one per AC.

    policy:
      - "longest": among AC:* candidates, pick longest seq
      - "first": first encountered
      - "lex": lexicographically smallest suffix after 'AC:' (if present), else exact
    """
    # Collect candidates per base AC
    exact: Dict[str, Tuple[str,str]] = {}          # AC -> (hdr, seq) when exact header '>AC'
    cand: Dict[str, List[Tuple[str,str]]] = defaultdict(list)  # AC -> list of AC:* candidates

    total = 0
    for hdr, seq in iter_fasta(fasta_path):
        total += 1
        tok = hdr[1:].strip().split()[0]
        base = tok.split(":", 1)[0] if ":" in tok else tok
        if base not in acs:
            continue
        if tok == base:
            # exact '>AC'
            # If multiple exacts exist (rare), keep the longest to be consistent
            if (base not in exact) or (len(seq) > len(exact[base][1])):
                exact[base] = (hdr, seq)
        else:
            cand[base].append((hdr, seq))

    # Resolve one per AC
    picked: List[Tuple[str,str]] = []
    stats = {
        "csv_ac_count": len(acs),
        "found_exact": 0,
        "found_from_candidates": 0,
        "ac_missing": 0,
        "ac_with_multiple_candidates": 0,
        "fasta_scanned": total,
    }

    for ac in sorted(acs):
        if ac in exact:
            picked.append(exact[ac])
            stats["found_exact"] += 1
            continue
        lst = cand.get(ac, [])
        if not lst:
            stats["ac_missing"] += 1
            continue

        if len(lst) > 1:
            stats["ac_with_multiple_candidates"] += 1

        if policy == "longest":
            hdr, seq = max(lst, key=lambda hs: len(hs[1]))
        elif policy == "first":
            hdr, seq = lst[0]
        elif policy == "lex":
            # sort by suffix (part after 'AC:' in the first token), then take first
            def suffix(hdr_seq):
                tok0 = hdr_seq[0][1:].strip().split()[0]
                suf = tok0.split(":", 1)[1] if ":" in tok0 else ""
                return suf
            hdr, seq = sorted(lst, key=suffix)[0]
        else:
            hdr, seq = max(lst, key=lambda hs: len(hs[1]))

        picked.append((hdr, seq))
        stats["found_from_candidates"] += 1

    return picked, stats

def main():
    p = argparse.ArgumentParser(description="Select exactly one FASTA record per AC from GraphPart CSV.")
    p.add_argument("--csv", required=True, help="GraphPart CSV (AC in first column or column named 'AC').")
    p.add_argument("--fasta", required=True, help="Original FASTA used for GraphPart.")
    p.add_argument("--out", required=True, help="Output FASTA (one record per AC).")
    p.add_argument("--policy", choices=["longest","first","lex"], default="longest",
                   help="If only AC:* candidates exist, how to choose. Default: longest.")
    args = p.parse_args()

    acs = load_acs(args.csv)
    print(f"Loaded ACs: {len(acs):,}")

    picked, stats = choose_representatives(acs, args.fasta, args.policy)
    n = write_fasta(picked, args.out)

    print(f"FASTA scanned:                    {stats['fasta_scanned']:,}")
    print(f"CSV ACs total:                    {stats['csv_ac_count']:,}")
    print(f"Exact matches written (>AC):      {stats['found_exact']:,}")
    print(f"Chosen from AC:* candidates:      {stats['found_from_candidates']:,}")
    print(f"ACs with multiple candidates:     {stats['ac_with_multiple_candidates']:,}")
    print(f"ACs not found in FASTA:           {stats['ac_missing']:,}")
    print(f"Wrote FASTA records:              {n:,}")
    print(f"Output:                           {args.out}")

if __name__ == "__main__":
    main()


'''
command python filter_fasta_by_graphpart_res.py \
  --csv ../../data/combined_datasets_graphpart_mmseqs.csv \
  --fasta ../../data/combined_l18_no_TMRs.fasta \
  --out ../../data/combined_l18_positive_final.fasta
  --policy first
'''