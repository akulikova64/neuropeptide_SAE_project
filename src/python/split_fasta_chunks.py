#!/usr/bin/env python3
# split_fasta_chunks.py
from pathlib import Path

FASTA_IN  = Path("../../data/novo_alldata_filter_out_enzymes/novo_alldata_no_enzymes_no_neuropeptides_max500.fasta")
OUT_DIR   = Path("../../data/deeptmhmm_l18_novo_alldata/chunks")
CHUNK_SIZE = 250  # DTU limit

OUT_DIR.mkdir(parents=True, exist_ok=True)

def iter_fasta(p: Path):
    h = None
    seq = []
    with p.open() as f:
        for line in f:
            if line.startswith(">"):
                if h is not None:
                    yield h, "".join(seq)
                h = line[1:].strip()
                seq = []
            else:
                seq.append(line.strip())
    if h is not None:
        yield h, "".join(seq)

def write_chunk(path: Path, records):
    with path.open("w") as w:
        for h, s in records:
            w.write(f">{h}\n")
            for i in range(0, len(s), 80):
                w.write(s[i:i+80] + "\n")

buf = []
chunk_idx = 1
total = 0
for hdr, seq in iter_fasta(FASTA_IN):
    buf.append((hdr, seq))
    total += 1
    if len(buf) == CHUNK_SIZE:
        out = OUT_DIR / f"chunk_{chunk_idx:04d}.fa"
        write_chunk(out, buf)
        print(f"Wrote {out} ({len(buf)} seqs)")
        buf = []
        chunk_idx += 1

# tail
if buf:
    out = OUT_DIR / f"chunk_{chunk_idx:04d}.fa"
    write_chunk(out, buf)
    print(f"Wrote {out} ({len(buf)} seqs)")

print(f"Total sequences split: {total}")
