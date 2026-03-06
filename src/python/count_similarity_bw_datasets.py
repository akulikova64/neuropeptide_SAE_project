#!/usr/bin/env python3
import os, sys, shutil, tempfile, subprocess, csv
from collections import defaultdict

# ========= USER PATHS =========
training_neuro_data   = "../../data/pph.fa"  # 32k neuropeptides from Thomas
uniprot_neuropeptides = "../../data/neuropeptide_sequences.fasta"
neuro_toxin_data      = "../../data/neuro_toxin_data_all.fasta"
insulins_data         = "../../data/allnr95_1_500_M_Reza_insulins.fasta"
filtered_secretome    = "../../data/secretome_filter_out_enzymes_5/secretome_no_enzymes.fasta"

# MMseqs thresholds
MIN_SEQ_ID = 0.40
COV        = 0.70
COV_MODE   = 1      # coverage on target
THREADS    = max(1, os.cpu_count() or 1)

# ========= FASTA HELPERS =========
def fasta_iter(path):
    """Yield (header, seq) tuples; header is the full '>' line without '>'."""
    header = None
    seq_chunks = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks).replace('\n','').replace('\r','').strip().upper()
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, ''.join(seq_chunks).replace('\n','').replace('\r','').strip().upper()

def fasta_size(path):
    """Return number of entries and a set of sequences (for exact-match checks)."""
    n = 0
    seqs = set()
    for _, s in fasta_iter(path):
        n += 1
        seqs.add(s)
    return n, seqs

# ========= MMSEQS HELPERS =========
def check_mmseqs():
    if shutil.which("mmseqs") is None:
        print("ERROR: mmseqs2 not found in PATH. Install or add to PATH.", file=sys.stderr)
        sys.exit(1)

def run(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    return res

def mmseqs_search_counts(query_fa, target_fa, workdir):
    """
    Run mmseqs createdb + search + convertalis and return:
      - homolog_pairs: number of unique (query_header, target_header) pairs
      - queries_with_hit: number of unique queries that hit at least one target
      - targets_with_hit: number of unique targets that were hit at least once
    """
    tmpdir = os.path.join(workdir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    qdb = os.path.join(workdir, "qDB")
    tdb = os.path.join(workdir, "tDB")
    adb = os.path.join(workdir, "alnDB")
    tsv = os.path.join(workdir, "hits.tsv")

    # createdb
    run(["mmseqs", "createdb", query_fa, qdb])
    run(["mmseqs", "createdb", target_fa, tdb])

    # search
    run([
        "mmseqs", "search", qdb, tdb, adb, tmpdir,
        "--threads", str(THREADS),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COV),
        "--cov-mode", str(COV_MODE),
        "-s", "7.5"
    ])

    # convert to TSV with headers that match FASTA >lines
    run([
        "mmseqs", "convertalis", qdb, tdb, adb, tsv,
        "--format-output", "qheader,theader,pident,alnlen,qlen,tlen,qcov,tcov"
    ])

    # parse TSV
    pairs = set()
    qset  = set()
    tset  = set()
    if os.path.exists(tsv):
        with open(tsv, 'r') as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2: 
                    continue
                qh, th = parts[0], parts[1]
                pairs.add((qh, th))
                qset.add(qh)
                tset.add(th)

    return len(pairs), len(qset), len(tset)

# ========= MAIN =========
def main():
    check_mmseqs()

    datasets = [
        ("UniProt neuropeptides", uniprot_neuropeptides),
        ("Neuropeps & toxins",    neuro_toxin_data),
        ("Insulins",              insulins_data),
        ("Filtered secretome",    filtered_secretome),
    ]
    train_label = "Training neuropeptides"
    train_path  = training_neuro_data

    # Basic size & exact-match sets
    print("[i] Scanning FASTA sizes and building exact-match sets…")
    n_train, train_seqs = fasta_size(train_path)
    sizes = {train_label: n_train}
    seqsets = {train_label: train_seqs}

    for label, path in datasets:
        n, seqs = fasta_size(path)
        sizes[label]  = n
        seqsets[label] = seqs

    # Prepare work root
    work_root = tempfile.mkdtemp(prefix="mmseqs_overlap_")
    print(f"[i] Temporary work dir: {work_root}")

    # Collect results
    rows = []
    for label, path in datasets:
        print(f"\n[comp] {train_label}  vs  {label}")
        # exact matches by sequence identity (byte-for-byte after whitespace removal, case-insensitive)
        exact = len(seqsets[train_label] & seqsets[label])
        print(f"    - exact sequence matches: {exact}")

        # mmseqs search (homologs)
        comp_dir = os.path.join(work_root, f"{label.replace(' ', '_')}")
        os.makedirs(comp_dir, exist_ok=True)
        homolog_pairs, q_with_hit, t_with_hit = mmseqs_search_counts(train_path, path, comp_dir)
        print(f"    - mmseqs close homologs (pairs): {homolog_pairs}")
        print(f"      queries with ≥1 hit: {q_with_hit}, targets with ≥1 hit: {t_with_hit}")

        rows.append({
            "comparison":         f"{train_label} vs {label}",
            "size_training":      sizes[train_label],
            "size_other":         sizes[label],
            "exact_matches":      exact,
            "homolog_pairs":      homolog_pairs,
            "queries_with_hit":   q_with_hit,
            "targets_with_hit":   t_with_hit
        })

    # Print a simple table to stdout
    print("\n=== Overlap summary ===")
    header = ["comparison", "size_training", "size_other", "exact_matches", "homolog_pairs", "queries_with_hit", "targets_with_hit"]
    colw = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in header}
    fmt = "  ".join("{:%d}" % colw[h] for h in header)
    print(fmt.format(*header))
    print("-" * (sum(colw.values()) + 2*(len(header)-1)))
    for r in rows:
        print(fmt.format(*(str(r[h]) for h in header)))

    # Also write CSV next to script
    out_csv = "mmseqs_overlap_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[✓] Wrote {out_csv}")

    # Cleanup message (leaving tmp for inspection; uncomment to auto-delete)
    print(f"[i] Temporary files kept at: {work_root}")
    # shutil.rmtree(work_root)

if __name__ == "__main__":
    main()
