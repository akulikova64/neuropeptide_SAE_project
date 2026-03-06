#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Config: input/output paths
###############################################################################
OUT_DIR="../../data/novo_alldata_filter_out_enzymes"

# Inputs from your previous enzyme-filter step
INPUT_FASTA="${OUT_DIR}/novo_alldata_no_enzymes.fasta"
INPUT_CSV="${OUT_DIR}/novo_alldata_filtered_nonenzymes_nocsdhit.csv"   # not strictly needed, but kept

# Workspace for this neuropeptide similarity step
NP_DIR="${OUT_DIR}/neuropeptide_filter"
mkdir -p "${NP_DIR}"

# Known neuropeptides (UniProt KW-0527)
NEURO_FASTA="${NP_DIR}/uniprot_kw0527_neuropeptides.fasta"
BLAST_DB_PREFIX="${NP_DIR}/neuropeptides_db"

# BLAST result files
HITS_RAW="${NP_DIR}/blast_hits_kw0527_raw.tsv"
HITS_FILT="${NP_DIR}/blast_hits_kw0527_filtered.tsv"

# Final flag list
OUT_FLAGS_CSV="${NP_DIR}/novo_neuropeptide_similarity_flags.csv"

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
    echo "⚠️  Warning: Input CSV not found or empty: ${INPUT_CSV}" >&2
    echo "    (CSV is not required for this flag-list script, continuing.)"
fi

###############################################################################
# 1) Download known neuropeptides (KW-0527) from UniProt as FASTA
###############################################################################
if [[ ! -s "${NEURO_FASTA}" ]]; then
    echo "[1/4] Downloading UniProt neuropeptides (KW-0527) …"
    curl -L \
      "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=keyword:KW-0527" \
      -o "${NEURO_FASTA}"
else
    echo "[1/4] Reusing existing UniProt neuropeptides FASTA → ${NEURO_FASTA}"
fi

###############################################################################
# 2) Build BLASTP database from known neuropeptides
###############################################################################
if [[ ! -s "${BLAST_DB_PREFIX}.pin" ]]; then
    echo "[2/4] Building BLAST protein database from UniProt neuropeptides …"
    makeblastdb -in "${NEURO_FASTA}" -dbtype prot -out "${BLAST_DB_PREFIX}" >/dev/null
else
    echo "[2/4] BLAST DB already exists → ${BLAST_DB_PREFIX}"
fi

###############################################################################
# 3) Run BLASTP: novo (query) vs neuropeptide DB (subject)
#    E-value ≤ 1e-10, then we apply:
#       - pident ≥ 80
#       - qcovs (query coverage) ≥ 70   <-- changed
###############################################################################
echo "[3/4] Running BLASTP (novo vs known neuropeptides) …"

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
awk '$3 >= 80 && $5 >= 70 {print}' "${HITS_RAW}" > "${HITS_FILT}" || true

if [[ ! -s "${HITS_FILT}" ]]; then
    echo "[3/4] No hits with pident ≥ 80% and qcovs ≥ 70%."
    echo "      Writing empty flag CSV (only header)."
    echo "Entry,percent_sim,closest_neurpeptide_entry,closest_neurpeptide_description" > "${OUT_FLAGS_CSV}"
    echo "✅ Done. Flag list: ${OUT_FLAGS_CSV}"
    exit 0
fi

###############################################################################
# 4) Build flag CSV:
#    For each query Entry, keep the best (highest pident) hit
#    Output columns:
#       Entry,percent_sim,closest_neurpeptide_entry,closest_neurpeptide_description
#
#    From NEURO_FASTA we derive, keyed by subject ID (sseqid token):
#       - closest_neurpeptide_entry (acc)
#       - closest_neurpeptide_description (full header minus '>')
###############################################################################
echo "[4/4] Building flag CSV (Entry, percent_sim, closest_neurpeptide_*) …"

{
    # header
    echo "Entry,percent_sim,closest_neurpeptide_entry,closest_neurpeptide_description"

    # body
    awk -v neuro="${NEURO_FASTA}" '
    BEGIN {
        #######################################################################
        # Parse neuropeptide FASTA to map:
        #   subj_id (raw token used by BLAST) ->
        #       subj_acc  (UniProt-like accession)
        #       subj_desc (full header minus ">")
        #######################################################################
        while ((getline line < neuro) > 0) {
            if (line ~ /^>/) {
                hdr = line
                sub(/^>/, "", hdr)  # remove ">"
                desc = hdr

                # first whitespace-separated token is the raw ID used by BLAST
                split(hdr, parts, " ")
                raw_id = parts[1]

                # Parse UniProt-like accession from raw_id:
                #   sp|ACC|NAME or tr|ACC|NAME -> ACC
                acc = raw_id
                n = split(raw_id, a, "|")
                if (n >= 3) {
                    acc = a[2]
                }
                sub(/-[0-9]+$/, "", acc)  # strip isoform suffix ACC-2 → ACC

                subj_acc[raw_id]  = acc
                subj_desc[raw_id] = desc
            }
        }
        close(neuro)
    }
    {
        q   = $1       # qseqid
        s   = $2       # sseqid (raw subject ID token from FASTA)
        pid = $3 + 0.0

        # --- parse Entry from qseqid ---
        qacc = q
        nq = split(q, qa, "|")
        if (nq >= 3) {
            qacc = qa[2]
        }
        sub(/-[0-9]+$/, "", qacc)

        # --- subject info from FASTA lookup ---
        sacc = subj_acc[s]
        sdesc = subj_desc[s]

        # fallback if not found in lookup (shouldn t really happen)
        if (sacc == "") {
            sacc = s
            ns = split(s, sa, "|")
            if (ns >= 3) {
                sacc = sa[2]
            }
            sub(/-[0-9]+$/, "", sacc)
        }
        if (sdesc == "") {
            sdesc = s
        }

        # keep best (highest) percent identity per Entry (qacc)
        if (!(qacc in best_pid) || pid > best_pid[qacc]) {
            best_pid[qacc]       = pid
            best_subj_acc[qacc]  = sacc
            best_subj_desc[qacc] = sdesc
        }
    }
    END {
        for (e in best_pid) {
            pid  = best_pid[e]
            acc  = best_subj_acc[e]
            desc = best_subj_desc[e]

            # escape double quotes in description and wrap desc in quotes
            gsub(/"/, "\"\"", desc)
            printf "%s,%.2f,%s,\"%s\"\n", e, pid, acc, desc
        }
    }
    ' "${HITS_FILT}" | sort -t',' -k2,2nr
} > "${OUT_FLAGS_CSV}"

echo "✅ Neuropeptide similarity flag list written to:"
echo "   • ${OUT_FLAGS_CSV}"
