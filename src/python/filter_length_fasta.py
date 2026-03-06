#!/usr/bin/env python3
import sys
from pathlib import Path

# Adjust these if your paths differ
INPUT_FASTA = Path("../../data/novo_alldata_filter_out_enzymes/novo_alldata_no_enzymes_no_neuropeptides.fasta")
OUTPUT_FASTA = Path("../../data/novo_alldata_filter_out_enzymes/novo_alldata_no_enzymes_no_neuropeptides_max500.fasta")

MAX_LEN = 500  # maximum allowed sequence length (aa)

def fasta_iter(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    with path.open() as fh:
        header = None
        seq_chunks = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)

def main():
    if not INPUT_FASTA.exists():
        sys.exit(f"❌ Input FASTA not found: {INPUT_FASTA}")

    n_total = 0
    n_kept = 0
    n_removed = 0

    with OUTPUT_FASTA.open("w") as out_fa:
        for header, seq in fasta_iter(INPUT_FASTA):
            n_total += 1
            if len(seq) <= MAX_LEN:
                n_kept += 1
                out_fa.write(f">{header}\n")
                # wrap lines at 80 chars
                for i in range(0, len(seq), 80):
                    out_fa.write(seq[i:i+80] + "\n")
            else:
                n_removed += 1

    print("✅ Length filtering complete")
    print(f"   • Input sequences:  {n_total}")
    print(f"   • Kept (≤ {MAX_LEN} aa): {n_kept}")
    print(f"   • Removed (> {MAX_LEN} aa): {n_removed}")
    print(f"   • Output FASTA: {OUTPUT_FASTA}")

if __name__ == "__main__":
    main()
