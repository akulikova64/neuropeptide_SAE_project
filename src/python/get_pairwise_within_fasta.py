#!/usr/bin/env python3
import csv
from Bio import SeqIO, pairwise2

# ─── Paths ────────────────────────────────────────────────────────────────
in_fasta   = "../../data/cleavage_site_sequences.fasta"
output_csv = "../../data/cleavage_site_sequences_pairwise.csv"

# ─── Load all records ──────────────────────────────────────────────────────
records = list(SeqIO.parse(in_fasta, "fasta"))

# ─── Open CSV for writing ─────────────────────────────────────────────────
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Entry_1", "Entry_2", "percent_seq_sim"])

    # ─── Pairwise alignments (each pair only once) ─────────────────────────
    n = len(records)
    for i in range(n):
        rec1 = records[i]
        # extract accession (between the '|' delimiters) or fallback to full id
        parts1 = rec1.id.split("|")
        id1 = parts1[1] if len(parts1) >= 2 else rec1.id
        seq1 = str(rec1.seq)

        for j in range(i + 1, n):
            rec2 = records[j]
            parts2 = rec2.id.split("|")
            id2 = parts2[1] if len(parts2) >= 2 else rec2.id
            seq2 = str(rec2.seq)

            # global identity‐only alignment
            aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
            aln_seq1, aln_seq2, score, start, end = aln

            # percent identity = matches / alignment length * 100
            percent = score / len(aln_seq1) * 100

            writer.writerow([id1, id2, f"{percent:.2f}"])

print(f"Done — results saved to {output_csv}")
