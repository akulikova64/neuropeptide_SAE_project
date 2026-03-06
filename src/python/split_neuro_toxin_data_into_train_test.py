#!/usr/bin/env python3
import os
import random
from math import ceil
from pathlib import Path
from Bio import SeqIO

# This script is going to enter all the homology folders and fasta file with one representative
# from each cluster and will take 10% of the data to set aside for testing.

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR       = Path("../../data/classifier_training_data_with_toxins")
HOMOLOGIES     = [40, 50, 60, 70, 80, 90]
SEED           = 42
TEST_FRACTION  = 0.10   # 10% of sequences
OUTPUT_ROOT    = Path("/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data")
TEST_ROOT      = OUTPUT_ROOT / "test_data"
TRAIN_ROOT     = OUTPUT_ROOT / "training_data"

random.seed(SEED)

for hom in HOMOLOGIES:
    # 1) locate the representative‐sequence FASTA
    hom_dir = BASE_DIR / f"classifier_training_data_mmseqs{hom}"
    fasta_fp = hom_dir / f"cluster{hom}_rep_seq.fasta"
    if not fasta_fp.exists():
        print(f"⚠️  Missing FASTA for homology {hom}: {fasta_fp}")
        continue

    # 2) read all entry IDs
    records = list(SeqIO.parse(str(fasta_fp), "fasta"))
    entries = [rec.id for rec in records]
    n_total = len(entries)
    if n_total == 0:
        print(f"⚠️  No sequences in {fasta_fp}")
        continue

    # 3) pick 10% for test (round up)
    n_test = max(1, ceil(n_total * TEST_FRACTION))
    test_entries = random.sample(entries, n_test)
    train_entries = [e for e in entries if e not in test_entries]

    # 4) prepare output folders
    test_dir  = TEST_ROOT    / f"homology_group{hom}"
    train_dir = TRAIN_ROOT   / f"homology_group{hom}"
    test_dir .mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # 5) write out space‐separated entry lists
    (test_dir / "test_entry_names.txt").write_text(" ".join(test_entries)  + "\n")
    (train_dir / "training_entry_names.txt").write_text(" ".join(train_entries) + "\n")

    print(f"✅ Homology {hom}: total={n_total}, test={n_test}, train={len(train_entries)}")
