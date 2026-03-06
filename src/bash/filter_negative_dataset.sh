# this is the filtering I used for the final nagative set
# where I sampled from 2M.


#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG — edit paths if needed
########################################
# ~2M secreted proteins minus neuropeptides:
MAIN_FASTA="/Volumes/T7 Shield/uniprot_secreted_minus_neuropeptides.fasta.gz"

# Positive dataset (to exclude if >0.4 id & >=0.7 cov):
POS_FASTA="../../data/combined_l18_no_TMRs.fasta"

# Secretome (no enzymes) dataset (to exclude if >0.8 id & >=0.7 cov):
SEC_FASTA="../../data/secretome_filter_out_enzymes_5/secretome_no_enzymes.fasta"

# Output on T7:
FINAL_FASTA="/Volumes/T7 Shield/filtered_negative_dataset_1.fasta"

# Threads for MMseqs2
THREADS="${THREADS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 8 )}"

# Temp working area
WRK="mmseqs_negset_$(date +%s)"
mkdir -p "$WRK"
TMP="$WRK/tmp"
mkdir -p "$TMP"

echo "Using THREADS=$THREADS"
echo "Working dir: $WRK"
echo

########################################
# Helper: tiny Python filter to drop IDs
########################################
cat > "$WRK/filter_fasta_exclude.py" << 'PY'
#!/usr/bin/env python3
import sys, gzip

in_path, exclude_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(exclude_path) as f:
    bad = set(x.strip() for x in f if x.strip())

def opener(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "rt")

def header_id(line):
    # Use token up to first whitespace
    return line[1:].strip().split()[0]

with opener(in_path) as fin, open(out_path, "w") as fout:
    keep = True
    cur_id = None
    for line in fin:
        if line.startswith(">"):
            cur_id = header_id(line)
            keep = (cur_id not in bad)
            if keep:
                fout.write(line)
        else:
            if keep:
                fout.write(line)
PY
chmod +x "$WRK/filter_fasta_exclude.py"

########################################
# STEP 0 — count input
########################################
echo "Counting records in MAIN_FASTA (this may take a moment)..."
if [[ "$MAIN_FASTA" == *.gz ]]; then
  TOTAL=$(zgrep -c '^>' "$MAIN_FASTA")
else
  TOTAL=$(grep -c '^>' "$MAIN_FASTA")
fi
echo "Total input sequences: $TOTAL"
echo

########################################
# STEP 1 — Remove sequences similar to POS_FASTA
#          (>0.4 identity, >=0.7 coverage on BOTH sides)
########################################
echo "STEP 1: Search vs POSITIVE (id>0.4, cov>=0.7 both) ..."
mmseqs createdb "$MAIN_FASTA"             "$WRK/mainDB"      >/dev/null
mmseqs createdb "$POS_FASTA"              "$WRK/posDB"       >/dev/null
mmseqs search   "$WRK/mainDB" "$WRK/posDB" "$WRK/out1" "$TMP" \
  --min-seq-id 0.4 -c 0.7 --cov-mode 2 -s 7 --threads "$THREADS" >/dev/null

mmseqs convertalis "$WRK/mainDB" "$WRK/posDB" "$WRK/out1" "$WRK/hits1.tsv" \
  --format-output "query,target,pident,qcov,tcov,evalue,alnlen" >/dev/null

# queries to drop (unique)
cut -f1 "$WRK/hits1.tsv" | sort -u > "$WRK/drop1.txt"
D1=$(wc -l < "$WRK/drop1.txt" || echo 0)
echo "Sequences to drop after STEP 1: $D1"

# Filter the FASTA (exclude drop1)
echo "Filtering FASTA after STEP 1..."
"$WRK/filter_fasta_exclude.py" "$MAIN_FASTA" "$WRK/drop1.txt" "$WRK/filtered1.fasta"

C1=$(grep -c '^>' "$WRK/filtered1.fasta")
echo "Remaining after STEP 1: $C1"
echo

########################################
# STEP 2 — Remove sequences similar to SEC_FASTA
#          (>0.8 identity, >=0.7 coverage on BOTH sides)
########################################
echo "STEP 2: Search vs SECRETOME-no-enzymes (id>0.8, cov>=0.7 both) ..."
mmseqs createdb "$WRK/filtered1.fasta"     "$WRK/filt1DB"   >/dev/null
mmseqs createdb "$SEC_FASTA"               "$WRK/secDB"     >/dev/null
mmseqs search   "$WRK/filt1DB" "$WRK/secDB" "$WRK/out2" "$TMP" \
  --min-seq-id 0.8 -c 0.7 --cov-mode 2 -s 7 --threads "$THREADS" >/dev/null

mmseqs convertalis "$WRK/filt1DB" "$WRK/secDB" "$WRK/out2" "$WRK/hits2.tsv" \
  --format-output "query,target,pident,qcov,tcov,evalue,alnlen" >/dev/null

cut -f1 "$WRK/hits2.tsv" | sort -u > "$WRK/drop2.txt"
D2=$(wc -l < "$WRK/drop2.txt" || echo 0)
echo "Sequences to drop after STEP 2: $D2"

# Final filter
echo "Filtering FASTA after STEP 2 (final set)..."
"$WRK/filter_fasta_exclude.py" "$WRK/filtered1.fasta" "$WRK/drop2.txt" "$FINAL_FASTA"

LEFT=$(grep -c '^>' "$FINAL_FASTA")
echo
echo "============================================================"
echo "Final sequences remaining: $LEFT (from initial $TOTAL)"
echo "Saved to: $FINAL_FASTA"
echo "============================================================"
