#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CONFIGURE
###############################################################################
#FASTA_IN="../../data/SAE_training/all_training_data.fasta"
FASTA_IN="../../data/cleavage_site_sequences.fasta"

# here I am getting 80% homology clusters so that I can sample my validation data
# all_training_data.fasta is the remaining peptides without the 500 set aside as the test dataset

# where to put everything (create the parent directory only!)
#OUTROOT="../../data/SAE_training_data_mmseqs80"       # parent folder
OUTROOT="../../data/classifier_training_data_mmseqs50"
PREFIX="$OUTROOT/cluster50"                           # file prefix
TMPDIR="$OUTROOT/tmp"                                 # scratch dir

REP_FASTA="$OUTROOT/final_rep_train_val_nr50.fa"                      # final reps
MIN_ID=0.5
THREADS=8

mkdir -p "$OUTROOT"
mkdir -p "$TMPDIR"

###############################################################################
# CLUSTER
###############################################################################
echo "▶︎ Clustering at 50 % PID …"
mmseqs easy-cluster  "$FASTA_IN"   "$PREFIX"   "$TMPDIR" \
       --min-seq-id $MIN_ID --cov-mode 0 --threads $THREADS

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
  if (!seen[$1]) {seen[$1]=1; members[$1]=$1}
  if ($1!=$2)    {members[$1]=members[$1] FS $2}
}
END {for (rep in members) print members[rep]}
' "$CLUST_TSV" | sort -k1,1 > "$MEMBERS_TSV"

echo "• Calculating cluster sizes …"
awk -F'\t' '{print $1 "\t" NF}' "$MEMBERS_TSV" | sort -k2,2nr > "$SIZES_TSV"

###############################################################################
# COUNTS
###############################################################################
orig=$(grep -c '^>' "$FASTA_IN")
nr80=$(grep -c '^>' "$REP_FASTA")

echo "────────────────────────────────────────────────────────"
echo "Original sequences : $orig"
echo "After 50 % filter  : $nr50"
echo "Number of clusters : $(wc -l < "$MEMBERS_TSV")"
echo "Cluster members    : $MEMBERS_TSV"
echo "Cluster sizes      : $SIZES_TSV"
echo "Done."
