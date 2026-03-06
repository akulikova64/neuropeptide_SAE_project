#!/usr/bin/env python3
import csv
from Bio import SeqIO
from Bio import pairwise2

# ─── Paths ────────────────────────────────────────────────────────────────
group2_fasta = "../../data/cleavage_site_sequences.fasta"
test_fasta   = "../../data/test_dataset_stats/clean_toxins.fasta"
output_csv   = "../../data/test_dataset_stats/toxin_pairwise_alignment_results.csv"

# ─── Load sequences ───────────────────────────────────────────────────────
group2_records = list(SeqIO.parse(group2_fasta, "fasta"))
test_records   = list(SeqIO.parse(test_fasta, "fasta"))

# ─── Open CSV for writing ─────────────────────────────────────────────────
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Entry_group_2",
        "Entry_test",
        "percent_seq_sim",
        "group_2_seq",
        "test_seq"
    ])

    # ─── Pairwise align every sequence in group_2 to every test sequence ───
    for rec1 in group2_records:
        parts1 = rec1.id.split("|")
        id1 = parts1[1] if len(parts1) >= 2 else rec1.id
        seq1 = str(rec1.seq)

        for rec2 in test_records:
            parts2 = rec2.id.split("|")
            id2 = parts2[1] if len(parts2) >= 2 else rec2.id
            seq2 = str(rec2.seq)

            # global alignment (identity only)
            aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
            aln_seq1, aln_seq2, score, start, end = aln

            # percent identity = matches / alignment length * 100
            percent = score / len(aln_seq1) * 100

            writer.writerow([
                id1,
                id2,
                f"{percent:.2f}",
                seq1,    # original group_2 sequence
                seq2     # original test sequence
            ])

print(f"Done — results saved to {output_csv}")
