#!/usr/bin/env python3
import re
from pathlib import Path

# Paths
#CHUNK_DIR  = Path("../../data/fig_6_mouse_secretome/secretome_filter_out_enzymes_5/deeptmhmm_chunks")
#RESULT_DIR = Path("../../data/fig_6_mouse_secretome/secretome_filter_out_enzymes_5/deeptmhmm_results")

#INPUT_FASTA  = Path("../../data/fig_6_mouse_secretome/secretome_filter_out_enzymes_5/secretome_no_immunoglobulins_deduplicated.fasta")
#OUTPUT_FASTA = Path("../../data/fig_6_mouse_secretome/secretome_filter_out_enzymes_5/secretome_no_TMRs.fasta")

CHUNK_DIR  = Path("../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5/deeptmhmm_chunks")
RESULT_DIR = Path("../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5/deeptmhmm_results")

INPUT_FASTA  = Path("../../data/fig_6_zebrafish_secretome/secretome_no_enzymes_deduplicated.fasta")
OUTPUT_FASTA = Path("../../data/fig_6_zebrafish_secretome/secretome_no_TMRs.fasta")

# ---- Helpers ----
def fasta_iter(path):
    """Yield (header, seq_string) with header = the whole >line minus '>' (no spaces stripped)."""
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).replace(" ", "").replace("\r", "").replace("\n", "").upper()
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks).replace(" ", "").replace("\r", "").replace("\n", "").upper()

def fasta_write(path, items):
    with open(path, "w") as out:
        for h, s in items:
            out.write(f">{h}\n")
            for i in range(0, len(s), 80):
                out.write(s[i:i+80] + "\n")

def header_token(h):
    """DeepTMHMM uses the first whitespace-separated token as the ID (e.g., 'pph_0')."""
    return h.split()[0]

def parse_gff3_for_tm_ids(gff_text):
    """
    Return set of IDs with Number of predicted TMRs > 0.
    GFF has blocks like:
      # pph_0 Length: 207
      # pph_0 Number of predicted TMRs: 0
    """
    tm_ids = set()
    current_id = None
    for line in gff_text.splitlines():
        line = line.strip()
        if line.startswith("# ") and "Length:" in line:
            m = re.match(r"#\s+(\S+)\s+Length:", line)
            if m:
                current_id = m.group(1)
        elif "Number of predicted TMRs:" in line and current_id:
            m = re.search(r"Number of predicted TMRs:\s*(\d+)", line)
            if m:
                n = int(m.group(1))
                if n > 0:
                    tm_ids.add(current_id)
    return tm_ids

# ---- 1) Collect TM IDs from all results ----
tm_ids_by_chunk = {}  # chunk_name -> set(ids like 'pph_123')
total_tm_ids = 0
for chunk_dir in sorted([p for p in RESULT_DIR.iterdir() if p.is_dir()]):
    gff = chunk_dir / f"{chunk_dir.name}.gff3"
    if not gff.exists():
        continue
    txt = gff.read_text(encoding="utf-8", errors="ignore")
    ids = parse_gff3_for_tm_ids(txt)
    tm_ids_by_chunk[chunk_dir.name] = ids
    total_tm_ids += len(ids)

print(f"[i] Parsed GFF3: {len(tm_ids_by_chunk)} chunk(s) with TM hits; total TM IDs = {total_tm_ids}")

# ---- 2) Map TM IDs to their sequences via chunk FASTAs ----
tm_seqs = set()
missing_ids = []
for chunk_name, ids in tm_ids_by_chunk.items():
    if not ids:
        continue
    # find a matching chunk fasta filename (chunk_XXXX.fa)
    chunk_fa = CHUNK_DIR / f"{chunk_name}.fa"
    if not chunk_fa.exists():
        # try alternative suffixes just in case
        candidates = list(CHUNK_DIR.glob(f"{chunk_name}.*"))
        found = False
        for c in candidates:
            if c.suffix.lower() in (".fa", ".fasta", ".faa"):
                chunk_fa = c
                found = True
                break
        if not found:
            print(f"[WARN] Could not find chunk FASTA for {chunk_name}")
            missing_ids.extend([(chunk_name, i) for i in ids])
            continue

    # build map ID -> sequence for this chunk
    id2seq = {}
    for h, s in fasta_iter(chunk_fa):
        id2seq[header_token(h)] = s

    # collect sequences for TM IDs
    for pid in ids:
        seq = id2seq.get(pid)
        if seq:
            tm_seqs.add(seq)
        else:
            missing_ids.append((chunk_name, pid))

print(f"[i] TM sequences collected: {len(tm_seqs)}")
if missing_ids:
    print(f"[WARN] {len(missing_ids)} TM IDs missing from chunk FASTAs (e.g., first 5): {missing_ids[:5]}")

# ---- 3) Filter the original combined FASTA by sequence ----
kept = []
removed = 0
total = 0
for h, s in fasta_iter(INPUT_FASTA):
    total += 1
    if s in tm_seqs:
        removed += 1
    else:
        kept.append((h, s))

OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
fasta_write(OUTPUT_FASTA, kept)

print("\n=== Summary ===")
print(f"Input combined FASTA: {INPUT_FASTA}")
print(f"Total sequences:      {total}")
print(f"Sequences removed (TM): {removed}")
print(f"Sequences kept:         {len(kept)}")
print(f"Output FASTA:         {OUTPUT_FASTA}")
