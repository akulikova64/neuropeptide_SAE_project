#!/usr/bin/env bash
set -euo pipefail

PRED_CSV="../../data/amphibian_analysis/amphibians_layer18_logreg_top320_predictions.csv"
FASTA="../../data/amphibian_analysis/amphibians_filter_out_enzymes_5/amphibians_deduplicated.fasta"
OUTDIR="../../data/amphibian_analysis"

OUT_90_100="${OUTDIR}/amphibians_bin_0.9_1.0.csv"
OUT_80_90="${OUTDIR}/amphibians_bin_0.8_0.9.csv"
OUT_70_80="${OUTDIR}/amphibians_bin_0.7_0.8.csv"

tmp_map="$(mktemp)"
trap 'rm -f "$tmp_map"' EXIT

# --- Build Entry<TAB>sequence map from FASTA (Entry = first token after '>') ---
awk '
BEGIN { entry=""; seq="" }
function flush() {
  if (entry != "") {
    gsub(/[ \t\r\n]/,"",seq)
    print entry "\t" seq
  }
}
/^>/{
  flush()
  entry=$0
  sub(/^>/,"",entry)
  split(entry,a,/[ \t]/)   # Entry is first token up to whitespace/tab
  entry=a[1]
  seq=""
  next
}
{
  gsub(/[ \t\r\n]/,"",$0)
  seq = seq $0
}
END { flush() }
' "$FASTA" > "$tmp_map"

# --- Join predictions with sequence map and write three CSVs ---
awk -v OUT1="$OUT_90_100" -v OUT2="$OUT_80_90" -v OUT3="$OUT_70_80" '
BEGIN {
  OFS=","
  print "Entry","probability","sequence" > OUT1
  print "Entry","probability","sequence" > OUT2
  print "Entry","probability","sequence" > OUT3
}

# Pass 1: load map (tab-separated) into seqmap[]
FNR==NR {
  split($0, a, "\t")
  seqmap[a[1]] = a[2]
  next
}

# Pass 2: read predictions CSV (comma-separated)
FNR==1 { next }  # skip header

function q(s,    t) { t=s; gsub(/"/, "\"\"", t); return "\"" t "\"" }

{
  # parse CSV line "Entry,probability" safely (Entry has no commas in your data)
  split($0, f, ",")
  entry = f[1]
  prob  = f[2] + 0.0

  seq = seqmap[entry]
  if (seq == "") next   # sequence not found -> skip

  if (prob >= 0.9 && prob <= 1.0) {
    print q(entry), sprintf("%.6f", prob), q(seq) >> OUT1
  } else if (prob >= 0.8 && prob < 0.9) {
    print q(entry), sprintf("%.6f", prob), q(seq) >> OUT2
  } else if (prob >= 0.7 && prob < 0.8) {
    print q(entry), sprintf("%.6f", prob), q(seq) >> OUT3
  }
}
' "$tmp_map" "$PRED_CSV"

echo "Wrote:"
echo "  $OUT_90_100"
echo "  $OUT_80_90"
echo "  $OUT_70_80"
