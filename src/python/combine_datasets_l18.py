# combining datasets

#!/usr/bin/env python3
import os, sys, shutil, tempfile, subprocess
from collections import OrderedDict

# -------------------- USER PATHS --------------------
training_neuro_data   = "../../data/pph.fa"  # 32k neuropeptides from Thomas
uniprot_neuropeptides = "../../data/neuropeptide_sequences.fasta"
hormones_data         = "../../data/OrthoDB_r2.fa"  # your “hormones_data”
filtered_secretome    = "../../data/secretome_filter_out_enzymes_5/secretome_no_enzymes.fasta"

# Outputs
out_combined_fasta    = "../../data/combined_training_l18.fasta"
out_filtered_fasta    = "../../data/combined_training_l18_noSecretomeHomologs.fasta"
out_hits_tsv          = "../../data/combined_vs_secretome_hits.tsv"  # keep evidence

# MMseqs thresholds for “too close” (i.e., sequences to exclude)
MIN_SEQ_ID = 0.80     # ≥80% identity
COV        = 0.70     # coverage threshold
COV_MODE   = 1        # coverage on target (secretome)
THREADS    = max(1, os.cpu_count() or 1)
# ---------------------------------------------------

def fasta_iter(path):
    """Yield (header, sequence) where header is the full >line without '>'."""
    header = None
    seq_chunks = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                if header is not None:
                    seq = ''.join(seq_chunks).replace('\n','').replace('\r','').strip().upper()
                    yield header, seq
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            seq = ''.join(seq_chunks).replace('\n','').replace('\r','').strip().upper()
            yield header, seq

def write_fasta(path, items):
    """items: iterable of (header, seq)"""
    with open(path, 'w') as out:
        for h, s in items:
            out.write(f">{h}\n")
            for i in range(0, len(s), 80):
                out.write(s[i:i+80] + "\n")

def check_mmseqs():
    if shutil.which("mmseqs") is None:
        print("ERROR: mmseqs2 not found in PATH. Install or add to PATH.", file=sys.stderr)
        sys.exit(1)

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    return r

def mmseqs_hits(query_fa, target_fa, workdir, out_tsv):
    """
    Run mmseqs search and return:
      hit_queries = set of query headers with ≥1 hit (to be excluded)
      n_pairs, n_q, n_t = counts for reporting
    Also writes a TSV with details to out_tsv.
    """
    tmpdir = os.path.join(workdir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    qdb = os.path.join(workdir, "qDB")
    tdb = os.path.join(workdir, "tDB")
    adb = os.path.join(workdir, "alnDB")
    tsv = os.path.join(workdir, "hits.tsv")

    run(["mmseqs", "createdb", query_fa, qdb])
    run(["mmseqs", "createdb", target_fa, tdb])

    run([
        "mmseqs", "search", qdb, tdb, adb, tmpdir,
        "--threads", str(THREADS),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COV),
        "--cov-mode", str(COV_MODE),
        "-s", "7.5"
    ])

    run([
        "mmseqs", "convertalis", qdb, tdb, adb, tsv,
        "--format-output", "qheader,theader,pident,alnlen,qlen,tlen,qcov,tcov"
    ])

    # Copy TSV to a user-visible location
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    shutil.copyfile(tsv, out_tsv)

    pairs = set()
    qset  = set()
    tset  = set()
    with open(tsv, 'r') as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2: 
                continue
            qh, th = cols[0], cols[1]
            pairs.add((qh, th))
            qset.add(qh)
            tset.add(th)

    return qset, len(pairs), len(qset), len(tset)

def main():
    print("[i] Loading inputs and deduplicating by exact amino-acid sequence…")
    sources = [
        ("training_neuro_data",   training_neuro_data),
        ("uniprot_neuropeptides", uniprot_neuropeptides),
        ("hormones_data",         hormones_data),
    ]

    input_sizes = {}
    combined_seq2hdr = OrderedDict()  # seq -> header (keep first seen)
    for label, path in sources:
        n = 0
        for hdr, seq in fasta_iter(path):
            n += 1
            if seq not in combined_seq2hdr:
                combined_seq2hdr[seq] = hdr
        input_sizes[label] = n
        print(f"  - {label}: {n} entries")

    # 1) Write combined FASTA (all unique)
    os.makedirs(os.path.dirname(out_combined_fasta), exist_ok=True)
    write_fasta(out_combined_fasta, ((h, s) for s, h in combined_seq2hdr.items()))
    print(f"[✓] Wrote combined unique set: {out_combined_fasta}")
    print(f"    Unique sequences: {len(combined_seq2hdr)}")

    # 2) Find combined sequences that are “too close” to secretome via MMseqs2
    print("\n[i] Detecting sequences too close to secretome (id ≥ 0.80, tcov ≥ 0.70)…")
    check_mmseqs()
    workdir = tempfile.mkdtemp(prefix="mmseqs_combined_vs_secretome_")
    print(f"    Working dir: {workdir}")

    hit_queries, n_pairs, n_q, n_t = mmseqs_hits(
        out_combined_fasta, filtered_secretome, workdir, out_hits_tsv
    )
    print(f"    MMseqs hits TSV: {out_hits_tsv}")
    print(f"    pairs={n_pairs}, query_seqs_with_hit={n_q}, target_seqs_hit={n_t}")

    # 3) Build a header -> sequence map for quick filtering
    hdr2seq = {hdr: seq for seq, hdr in combined_seq2hdr.items()}

    # 4) Filter out any combined sequence whose header is in hit_queries
    kept = []
    removed = 0
    for hdr, seq in hdr2seq.items():
        if hdr in hit_queries:
            removed += 1
        else:
            kept.append((hdr, seq))

    # 5) Write filtered FASTA (no secretome-close homologs)
    write_fasta(out_filtered_fasta, kept)
    print(f"[✓] Wrote filtered (no secretome homologs) FASTA: {out_filtered_fasta}")
    print(f"    Kept: {len(kept)}  |  Removed (too close to secretome): {removed}")

    # 6) Summary
    print("\n=== Summary ===")
    for k, v in input_sizes.items():
        print(f"  {k:>24}: {v}")
    print(f"Combined unique sequences            : {len(combined_seq2hdr)}  ({out_combined_fasta})")
    print(f"Removed due to ≥80% id, tcov ≥0.70  : {removed}")
    print(f"Filtered unique sequences (kept)     : {len(kept)}  ({out_filtered_fasta})")
    print(f"MMseqs evidence TSV                  : {out_hits_tsv}")
    print(f"[i] Temporary files kept at: {workdir}  (delete when satisfied)")

if __name__ == "__main__":
    main()
