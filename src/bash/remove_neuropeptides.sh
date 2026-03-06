#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Config: input/output paths
###############################################################################
OUT_DIR="../../data/novo_alldata_filter_out_enzymes"

# Inputs from your previous enzyme-filter step
INPUT_FASTA="${OUT_DIR}/novo_alldata_no_enzymes.fasta"
INPUT_CSV="${OUT_DIR}/novo_alldata_filtered_nonenzymes_nocsdhit.csv"

# Workspace for this neuropeptide-filter step
NP_DIR="${OUT_DIR}/neuropeptide_filter"
mkdir -p "${NP_DIR}"

# Known neuropeptides (UniProt KW-0527)
NEURO_FASTA="${NP_DIR}/uniprot_kw0527_neuropeptides.fasta"
BLAST_DB_PREFIX="${NP_DIR}/neuropeptides_db"

# BLAST result files
HITS_RAW="${NP_DIR}/blast_hits_kw0527_raw.tsv"
HITS_FILT="${NP_DIR}/blast_hits_kw0527_filtered.tsv"

# Lists of IDs to remove
REMOVE_IDS="${NP_DIR}/secretome_ids_to_remove.txt"     # FASTA tokens (qseqid)
REMOVE_ENTRIES="${NP_DIR}/entries_to_remove.txt"       # UniProt accessions (for CSV)

# Final outputs
OUT_FASTA_FILTERED="${OUT_DIR}/novo_alldata_no_enzymes_no_neuropeptides.fasta"
OUT_CSV_FILTERED="${OUT_DIR}/novo_alldata_filtered_nonenzymes_nocsdhit_no_neuropeptides.csv"
OUT_FASTA_REMOVED="${OUT_DIR}/novo_alldata_too_close_to_neuropeptides.fasta"

###############################################################################
# Check dependencies and inputs
###############################################################################
for exe in curl blastp makeblastdb awk; do
    if ! command -v "${exe}" >/dev/null 2>&1; then
        echo "❌ Error: required executable '${exe}' not found in PATH." >&2
        exit 1
    fi
done

if [[ ! -s "${INPUT_FASTA}" ]]; then
    echo "❌ Input FASTA not found or empty: ${INPUT_FASTA}" >&2
    exit 1
fi

if [[ ! -s "${INPUT_CSV}" ]]; then
    echo "❌ Input CSV not found or empty: ${INPUT_CSV}" >&2
    exit 1
fi

###############################################################################
# 1) Download known neuropeptides (KW-0527) from UniProt as FASTA
###############################################################################
if [[ ! -s "${NEURO_FASTA}" ]]; then
    echo "[1/5] Downloading UniProt neuropeptides (KW-0527) …"
    curl -L \
      "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=keyword:KW-0527" \
      -o "${NEURO_FASTA}"
else
    echo "[1/5] Reusing existing UniProt neuropeptides FASTA → ${NEURO_FASTA}"
fi

###############################################################################
# 2) Build BLASTP database from known neuropeptides
###############################################################################
if [[ ! -s "${BLAST_DB_PREFIX}.pin" ]]; then
    echo "[2/5] Building BLAST protein database from UniProt neuropeptides …"
    makeblastdb -in "${NEURO_FASTA}" -dbtype prot -out "${BLAST_DB_PREFIX}" \
        >/dev/null
else
    echo "[2/5] BLAST DB already exists → ${BLAST_DB_PREFIX}"
fi

###############################################################################
# 3) Run BLASTP: novo (query) vs neuropeptide DB (subject)
#    E-value ≤ 1e-10, then we apply:
#       - pident ≥ 95
#       - qcovs (query coverage) ≥ 90
###############################################################################
echo "[3/5] Running BLASTP (novo vs known neuropeptides) …"

THREADS="$(nproc 2>/dev/null || echo 1)"

blastp \
  -query "${INPUT_FASTA}" \
  -db "${BLAST_DB_PREFIX}" \
  -evalue 1e-10 \
  -outfmt "6 qseqid sseqid pident length qcovs evalue qlen slen" \
  -num_threads "${THREADS}" \
  > "${HITS_RAW}"

echo "   • BLAST raw hits written to ${HITS_RAW}"

# Filter hits by identity and query coverage
# Columns: 1=qseqid, 2=sseqid, 3=pident, 4=aln length, 5=qcovs, 6=evalue, 7=qlen, 8=slen
awk '$3 >= 95 && $5 >= 90 {print}' "${HITS_RAW}" > "${HITS_FILT}" || true

if [[ ! -s "${HITS_FILT}" ]]; then
    echo "[3/5] No hits with pident ≥ 95% and qcovs ≥ 90%. Nothing to remove."
    # Just copy inputs to outputs and exit
    cp "${INPUT_FASTA}" "${OUT_FASTA_FILTERED}"
    cp "${INPUT_CSV}" "${OUT_CSV_FILTERED}"
    # No removed FASTA because nothing was removed
    : > "${OUT_FASTA_REMOVED}"
    echo "✅ Done. Filtered FASTA: ${OUT_FASTA_FILTERED}"
    echo "✅ Done. Filtered CSV:  ${OUT_CSV_FILTERED}"
    exit 0
fi

# Get unique query IDs (secretome sequence IDs) to remove
cut -f1 "${HITS_FILT}" | sort -u > "${REMOVE_IDS}"

N_REMOVE_IDS="$(wc -l < "${REMOVE_IDS}" | tr -d '[:space:]')"
echo "   • Query sequences flagged for removal (neuropeptide-like): ${N_REMOVE_IDS}"

###############################################################################
# 4) Convert FASTA tokens → UniProt accessions (for CSV filtering)
#    Logic: token is first whitespace-separated header token:
#       - If it looks like sp|ACC|NAME or tr|ACC|NAME → use ACC (middle part)
#       - Otherwise use the token itself
#       - Strip isoform suffix: ACC-2 → ACC
###############################################################################
echo "[4/5] Converting FASTA IDs to UniProt accessions for CSV filtering …"

awk '
{
    raw = $1
    acc = raw
    # Split on pipe: sp|ACC|NAME or tr|ACC|NAME
    n = split(raw, arr, "|")
    if (n >= 3) {
        acc = arr[2]
    }
    # Strip isoform suffix: ACC-2 → ACC
    sub(/-[0-9]+$/, "", acc)
    print acc
}
' "${REMOVE_IDS}" | sort -u > "${REMOVE_ENTRIES}"

N_REMOVE_ENTRIES="$(wc -l < "${REMOVE_ENTRIES}" | tr -d '[:space:]')"
echo "   • Unique UniProt-like accessions to remove from CSV: ${N_REMOVE_ENTRIES}"

###############################################################################
# 5) Write:
#    - FASTA with removed sequences
#    - FASTA with remaining sequences
#    - CSV with remaining entries
###############################################################################
echo "[5/5] Writing filtered FASTA/CSV outputs …"

# 5a) FASTA of removed sequences (too close to known neuropeptides)
awk -v rm="${REMOVE_IDS}" '
BEGIN {
    while ((getline line < rm) > 0) {
        to_rm[line] = 1
    }
    close(rm)
}
/^>/ {
    hdr = substr($0, 2)           # strip ">"
    split(hdr, parts, " ")
    id = parts[1]                 # first token is qseqid
    remove = (id in to_rm)
}
{
    if (remove) print
}
' "${INPUT_FASTA}" > "${OUT_FASTA_REMOVED}"

# 5b) FASTA of remaining sequences
awk -v rm="${REMOVE_IDS}" '
BEGIN {
    while ((getline line < rm) > 0) {
        to_rm[line] = 1
    }
    close(rm)
}
/^>/ {
    hdr = substr($0, 2)
    split(hdr, parts, " ")
    id = parts[1]
    keep = !(id in to_rm)
}
{
    if (keep) print
}
' "${INPUT_FASTA}" > "${OUT_FASTA_FILTERED}"

# 5c) CSV of remaining entries (single-column CSV: "Entry")
#     We remove rows whose Entry matches any in entries_to_remove.txt
head -n 1 "${INPUT_CSV}" > "${OUT_CSV_FILTERED}"   # keep header

tail -n +2 "${INPUT_CSV}" | awk -F',' -v rm="${REMOVE_ENTRIES}" '
BEGIN {
    while ((getline line < rm) > 0) {
        drop[line] = 1
    }
    close(rm)
}
{
    if (!($1 in drop)) print
}
' >> "${OUT_CSV_FILTERED}"

echo
echo "✅ Neuropeptide-filter step completed."
echo "   • Removed sequences FASTA:       ${OUT_FASTA_REMOVED}"
echo "   • Remaining sequences FASTA:     ${OUT_FASTA_FILTERED}"
echo "   • Remaining entries CSV:         ${OUT_CSV_FILTERED}"
