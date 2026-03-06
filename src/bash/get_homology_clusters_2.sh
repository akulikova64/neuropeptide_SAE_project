#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CONFIGURE
###############################################################################
# Homology cutoff in percent (e.g. 50, 70, 90)
HOM=90
# convert to decimal for mmseqs’s --min-seq-id
MIN_ID=$(awk "BEGIN { printf \"%.2f\", ${HOM}/100 }")

#FASTA_IN="../../data/cleavage_site_sequences.fasta"
FASTA_IN="../../data/neuro_toxin_data_all.fasta"

# output root & prefix, based on cutoff
OUTROOT="../../data/classifier_training_data_with_toxins/classifier_training_data_mmseqs${HOM}"
PREFIX="$OUTROOT/cluster${HOM}"
TMPDIR="$OUTROOT/tmp"

# final rep fasta
REP_FASTA="$OUTROOT/final_rep_train_val_nr${HOM}.fasta"
THREADS=8

mkdir -p "$OUTROOT" "$TMPDIR"

###############################################################################
# CLUSTER
###############################################################################
echo "▶︎ Clustering at ${HOM}% PID …"
mmseqs easy-cluster \
    "$FASTA_IN" \
    "$PREFIX" \
    "$TMPDIR" \
    --min-seq-id $MIN_ID \
    --cov-mode 0 \
    --threads $THREADS

###############################################################################
# COPY REPRESENTATIVES
###############################################################################
cp "${PREFIX}_rep_seq.fasta" "$REP_FASTA"
echo "• Representatives → $REP_FASTA"

###############################################################################
# ONE-LINE-PER-CLUSTER TABLE
###############################################################################
CLUST_TSV="${PREFIX}_cluster.tsv"
MEMBERS_TSV="$OUTROOT/cluster_members.tsv"
SIZES_TSV="$OUTROOT/cluster_sizes.tsv"

echo "• Building cluster member list …"
awk -F'\t' '
{
  if (!seen[$1]) { seen[$1]=1; members[$1]=$1 }
  if ($1 != $2)  { members[$1]=members[$1] FS $2 }
}
END {
  for (rep in members) print members[rep]
}
' "$CLUST_TSV" | sort -k1,1 > "$MEMBERS_TSV"

echo "• Calculating cluster sizes …"
awk -F'\t' '{ print $1 "\t" NF }' "$MEMBERS_TSV" | sort -k2,2nr > "$SIZES_TSV"

###############################################################################
# COUNTS
###############################################################################
orig=$(grep -c '^>' "$FASTA_IN")
nr=$(grep -c '^>' "$REP_FASTA")

echo "────────────────────────────────────────────────────────"
echo "Original sequences : $orig"
echo "After ${HOM}% filter : $nr"
echo "Number of clusters  : $(wc -l < "$MEMBERS_TSV")"
echo "Cluster members     : $MEMBERS_TSV"
echo "Cluster sizes       : $SIZES_TSV"
echo "Done."
