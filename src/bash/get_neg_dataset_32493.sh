#!/usr/bin/env bash
set -euo pipefail

### ---------------- USER SETTINGS ----------------
NP_FASTA="/Volumes/T7 Shield/neuropep_training_SAE.fasta"   # path to your neuropeptide FASTA
OUTDIR="/Users/anastasiyakulikova/Desktop/SAE_project/data/negatives_build_matched"
THREADS=16
LENGTH_CSV="../../data/length_bins_50_training_data.csv"    # <-- your 50-aa bin CSV

# MMseqs2 similarity thresholds
MIN_SEQ_ID=0.40       # ≥40% identity
COV=0.70              # coverage threshold (fraction)
COV_MODE=1            # coverage on target sequence

# Sampling target
TARGET_TOTAL=32492
SEED=13
### ------------------------------------------------

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# macOS: allow many open files
ulimit -n 65536 || true

# Deps
command -v mmseqs >/dev/null 2>&1 || { echo "mmseqs2 not found in PATH"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "curl not found"; exit 1; }
[[ -f "$LENGTH_CSV" ]] || { echo "Length CSV not found: $LENGTH_CSV" >&2; exit 1; }
python - <<'PY' >/dev/null 2>&1 || { echo "Python missing?"; exit 1; }
import sys
PY

echo "[+] Step 1/6: Download 250,000 secreted UniProt entries (not limited to Swiss-Prot)…"
# Temporarily relax pipefail so curl's SIGPIPE on early exit doesn't kill the script
set +o pipefail
curl -sS -L --fail --retry 5 --retry-delay 2 \
  "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=keyword:KW-0964" \
| awk -v max=250000 '
    BEGIN{c=0; print_flag=0}
    /^>/ { c++; print_flag = (c<=max) }
    { if (print_flag) print; if (c>max) exit }
  ' > secreted_kw0964_200k.fasta
set -o pipefail
echo "[+] Done: $(grep -c '^>' secreted_kw0964_200k.fasta) sequences saved"

echo "[+] Step 2/6: Remove any exact neuropeptide headers (belt & suspenders)…"
grep '^>' "$NP_FASTA" | awk '{tok=$1; sub(/^>/,"",tok); print tok}' | sort -u > np_ids.txt
awk 'BEGIN{while((getline<"np_ids.txt")>0){drop[$0]=1}}
     /^>/{tok=$1; sub(/^>/,"",tok); keep=!drop[tok]}
     {if(keep) print}' \
  secreted_kw0964_200k.fasta > secreted_kw0964_200k_noExactNP.fasta
echo "[+] After exact-drop: $(grep -c '^>' secreted_kw0964_200k_noExactNP.fasta) remain"

echo "[+] Step 3/6: Build MMseqs2 DBs…"
rm -rf npDB* secDB* tmp_search alnDB* hits.tsv || true
mmseqs createdb "$NP_FASTA" npDB
mmseqs createdb secreted_kw0964_200k_noExactNP.fasta secDB

echo "[+] Step 4/6: Search for sequences similar to neuropeptides…"
mmseqs search npDB secDB alnDB tmp_search \
  --threads "$THREADS" -s 7.5 --min-seq-id "$MIN_SEQ_ID" -c "$COV" --cov-mode "$COV_MODE"

# Use headers so we can match FASTA IDs exactly
mmseqs convertalis npDB secDB alnDB hits.tsv \
  --format-output "qheader,theader,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,qlen,tlen"

echo "[+] Step 5/6: Drop similar targets and build the eligible pool…"
# Targets to drop: first token of theader
awk -F'\t' '{print $2}' hits.tsv | awk '{print $1}' | sort -u > too_close_ids.txt
cat np_ids.txt too_close_ids.txt | sort -u > drop_ids.txt
echo "[i] Will drop $(wc -l < drop_ids.txt | tr -d " ") sequences due to similarity/exact match"

awk 'BEGIN{while((getline<"drop_ids.txt")>0){drop[$0]=1}}
     /^>/{tok=$1; sub(/^>/,"",tok); keep=!drop[tok]}
     {if(keep) print}' \
  secreted_kw0964_200k_noExactNP.fasta > secreted_filtered_not_similar_to_NP.fasta

ELIG=$(grep -c '^>' secreted_filtered_not_similar_to_NP.fasta || echo 0)
echo "[+] Eligible pool after similarity filtering: $ELIG sequences"

echo "[+] Step 6/6: Stratified sampling to match CSV length distribution (total: $TARGET_TOTAL)…"
python - "$LENGTH_CSV" "$TARGET_TOTAL" "$SEED" <<'PY'
import sys, re, random

csv_path = sys.argv[1]
target_total = int(sys.argv[2])
seed = int(sys.argv[3])
random.seed(seed)

# Parse desired distribution (CSV with header: bin,count where bin like "0-50","51-100",...)
desired = []  # list of (label, count)
with open(csv_path) as f:
    header = f.readline()
    for line in f:
        line=line.strip()
        if not line: continue
        b, c = line.split(",")
        b = b.strip()
        c = int(c.strip())
        if c>0:
            desired.append((b,c))

# Verify total
tot = sum(c for _,c in desired)
if tot != target_total:
    print(f"[WARN] CSV total {tot} != requested {target_total}. Will use CSV total = {tot}.", file=sys.stderr)
    target_total = tot

# Helper: bin label from length using the same scheme as the CSV (0-50, 51-100, 101-150, ...)
def bin_label(L):
    if L <= 50:
        s,e = 0,50
    else:
        BIN = 50
        idx = 1 + (L-51)//BIN
        s = 51 + (idx-1)*BIN
        e = s + BIN - 1
    return f"{s}-{e}"

# Read eligible FASTA and bucket IDs by bin
bins = {}  # label -> [header_line(s)]
with open("secreted_filtered_not_similar_to_NP.fasta") as f:
    hdr = None
    L = 0
    seq_lines = []
    for line in f:
        if line.startswith(">"):
            # flush previous
            if hdr is not None:
                lbl = bin_label(L)
                bins.setdefault(lbl, []).append((hdr, "".join(seq_lines)))
            hdr = line.rstrip("\n")
            L = 0
            seq_lines = []
        else:
            s = line.strip()
            if s:
                L += len(s)
                seq_lines.append(s+"\n")
    if hdr is not None:
        lbl = bin_label(L)
        bins.setdefault(lbl, []).append((hdr, "".join(seq_lines)))

# Sample per bin according to desired counts
out_path = "secreted_non_neuropeptides_32492_matched.fasta"
picked_total = 0
short_bins = []
with open(out_path, "w") as out:
    for lbl, need in desired:
        have = len(bins.get(lbl, []))
        if have < need:
            short_bins.append((lbl, need, have))
            # take all we have to keep going (or choose to fail)
            chosen = bins.get(lbl, [])
        else:
            chosen = random.sample(bins[lbl], need)
        for hdr, seq in chosen:
            out.write(hdr + "\n")
            out.write(seq)
        picked_total += len(chosen)

# Report
print(f"[OK] Wrote {out_path} with {picked_total} sequences.")
if short_bins:
    print("[WARN] Some bins had fewer eligible sequences than requested:", file=sys.stderr)
    for lbl, need, have in short_bins:
        print(f"  bin {lbl}: need {need}, have {have}", file=sys.stderr)
PY

echo "[✓] Final output: $OUTDIR/secreted_non_neuropeptides_32492_matched.fasta"

# Sanity: show total and a quick header count
grep -c '^>' secreted_non_neuropeptides_32492_matched.fasta || true
