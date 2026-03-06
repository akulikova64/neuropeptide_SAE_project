#!/usr/bin/env python3
"""
Match negative dataset sequence-length distribution to the positive dataset.

Steps:
1) Load positive FASTA and compute 50-aa length-bin counts.
2) Save positive bin counts to ../../data/combined_pos_data_size_dist.csv
3) Load negative FASTA and bin sequences.
4) For each bin, sample the same count (or as many as available) using seed=9427385.
5) Write the matched negative set to ../.../data/negative_dataset_matching_dist.fasta

Bin definition:
- 50-aa increments: [0–49], [50–99], [100–149], ...
- A sequence of length L goes to bin_start = 50 * ((L-1)//50), label "start-end"

Usage (paths are hardcoded to the ones you gave, but can be overridden by flags if desired):
    python make_matched_negative_set.py
"""

import os
import sys
import gzip
import argparse
import random
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Dict, List

# ---------- Defaults from your request ----------
DEFAULT_POS_FASTA  = "../../data/combined_l18_positive_final.fasta"
DEFAULT_NEG_FASTA  = "../../data/negative_dataset_matching_dist.fasta"
DEFAULT_POS_CSV    = "../../data/postive_graphpart_data_binned.csv"
DEFAULT_OUT_FASTA  = "../../data/negative_l18_dataset_final.fasta"
DEFAULT_BIN_SIZE   = 50
DEFAULT_SEED       = 9437569


def open_auto(path: str, mode: str = "rt"):
    """Open plain or gzipped files by extension."""
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header_line, sequence_string) from a FASTA."""
    with open_auto(path, "rt") as fh:
        header = None
        seq_chunks = []
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).replace("\n", "").replace("\r", "")
                header = line.strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks).replace("\n", "").replace("\r", "")


def bin_label(length: int, bin_size: int = DEFAULT_BIN_SIZE) -> str:
    """Return a human-readable bin label like '0-49', '50-99', ..."""
    if length <= 0:
        start = 0
    else:
        start = bin_size * ((length - 1) // bin_size)
    end = start + bin_size - 1
    return f"{start}-{end}"


def count_bins_from_fasta(path: str, bin_size: int) -> Counter:
    """Count number of sequences per 50-aa bin."""
    counts = Counter()
    for _, seq in iter_fasta(path):
        L = len(seq)
        lbl = bin_label(L, bin_size)
        counts[lbl] += 1
    return counts


def write_pos_distribution_csv(counts: Counter, out_csv: str, bin_size: int):
    """Write CSV with columns: bin,count,start,end."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # Make ordered rows by start index
    def start_of(lbl: str) -> int:
        return int(lbl.split("-")[0])
    rows = sorted(((lbl, cnt) for lbl, cnt in counts.items()), key=lambda x: start_of(x[0]))

    with open(out_csv, "w") as f:
        f.write("bin,count,start,end\n")
        for lbl, cnt in rows:
            start, end = lbl.split("-")
            f.write(f"{lbl},{cnt},{start},{end}\n")


def load_negative_buckets(path: str, bin_size: int) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a dict: bin_label -> list of (header, seq).
    For later sampling per bin.
    """
    buckets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for hdr, seq in iter_fasta(path):
        L = len(seq)
        lbl = bin_label(L, bin_size)
        buckets[lbl].append((hdr, seq))
    return buckets


def sample_per_bin(
    pos_counts: Counter,
    neg_buckets: Dict[str, List[Tuple[str, str]]],
    seed: int
) -> List[Tuple[str, str]]:
    """
    For each bin in positive counts, sample the same number from negative buckets (without replacement).
    If not enough in a bin, take as many as available and note a shortfall.
    Returns a list of (header, seq) sampled from negative set.
    """
    rnd = random.Random(seed)
    sampled: List[Tuple[str, str]] = []
    shortfalls = []

    # Process bins in ascending order of start for readability
    def start_of(lbl: str) -> int:
        return int(lbl.split("-")[0])
    for lbl in sorted(pos_counts.keys(), key=start_of):
        need = pos_counts[lbl]
        have_list = neg_buckets.get(lbl, [])
        have = len(have_list)
        if have == 0:
            shortfalls.append((lbl, need, 0))
            continue
        take = min(need, have)
        # sample without replacement
        if take == have:
            chosen = have_list  # all items
        else:
            chosen = rnd.sample(have_list, take)
        sampled.extend(chosen)
        if take < need:
            shortfalls.append((lbl, need, take))

    # Report shortfalls (printed to stderr-like info)
    if shortfalls:
        sys.stderr.write("NOTE: Not enough negatives in some bins — downsampled these bins:\n")
        for lbl, need, took in shortfalls:
            sys.stderr.write(f"  Bin {lbl}: need {need}, took {took}\n")

    return sampled


def write_fasta(records: List[Tuple[str, str]], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for hdr, seq in records:
            # hdr already starts with '>'
            f.write(f"{hdr}\n")
            # wrap sequence to 60 chars per line for readability
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


def main():
    ap = argparse.ArgumentParser(description="Match negative dataset length distribution to positive dataset (50-aa bins).")
    ap.add_argument("--pos_fasta", default=DEFAULT_POS_FASTA, help="Path to positive FASTA")
    ap.add_argument("--neg_fasta", default=DEFAULT_NEG_FASTA, help="Path to negative FASTA")
    ap.add_argument("--pos_csv",   default=DEFAULT_POS_CSV, help="Output CSV for positive size distribution")
    ap.add_argument("--out_fasta", default=DEFAULT_OUT_FASTA, help="Output FASTA for matched negative dataset")
    ap.add_argument("--bin_size",  type=int, default=DEFAULT_BIN_SIZE, help="Bin size (aa), default 50")
    ap.add_argument("--seed",      type=int, default=DEFAULT_SEED, help="Random seed for sampling")
    args = ap.parse_args()

    # 1) Positive distribution
    print(f"Reading positive FASTA: {args.pos_fasta}")
    pos_counts = count_bins_from_fasta(args.pos_fasta, args.bin_size)
    pos_total = sum(pos_counts.values())
    print(f"Positive sequences total: {pos_total}")
    write_pos_distribution_csv(pos_counts, args.pos_csv, args.bin_size)
    print(f"Wrote positive size distribution → {args.pos_csv}")

    # 2) Negative buckets
    print(f"Reading negative FASTA: {args.neg_fasta}")
    neg_buckets = load_negative_buckets(args.neg_fasta, args.bin_size)
    neg_total = sum(len(v) for v in neg_buckets.values())
    print(f"Negative sequences available: {neg_total}")

    # 3) Sample per bin
    sampled = sample_per_bin(pos_counts, neg_buckets, args.seed)
    print(f"Sampled negatives total: {len(sampled)}")

    # 4) Write output FASTA (ensure .fasta extension)
    out_fa = args.out_fasta if args.out_fasta.endswith(".fasta") else args.out_fasta + ".fasta"
    write_fasta(sampled, out_fa)
    print(f"Wrote matched negative FASTA → {out_fa}")

    # 5) Quick summary of matched distribution (optional print)
    matched_counts = Counter(bin_label(len(seq), args.bin_size) for _, seq in sampled)
    # Show first 10 bins for brevity
    def start_of(lbl: str) -> int:
        return int(lbl.split("-")[0])
    preview = sorted(matched_counts.items(), key=lambda x: start_of(x[0]))[:10]
    if preview:
        print("Preview matched distribution (first 10 bins):")
        for lbl, cnt in preview:
            need = pos_counts.get(lbl, 0)
            print(f"  {lbl}: sampled={cnt}, positive_need={need}")


if __name__ == "__main__":
    main()
