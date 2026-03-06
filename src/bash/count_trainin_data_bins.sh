#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-"../../data/combined_l18_positive_final.fasta"}
OUTCSV=${2:-"../../data/postive_graphpart_data_binned.csv"}

if [[ ! -f "$INPUT" ]]; then
  echo "Input FASTA not found: $INPUT" >&2
  exit 1
fi

awk -v BIN=50 '
  BEGIN { L=0; maxb=0 }
  /^>/ {
    if (L>0) {
      if (L<=50) b=0; else b=1+int((L-51)/BIN);
      c[b]++; if (b>maxb) maxb=b
    }
    L=0; next
  }
  {
    gsub(/[[:space:]]/,""); L += length($0)
  }
  END {
    if (L>0) {
      if (L<=50) b=0; else b=1+int((L-51)/BIN);
      c[b]++; if (b>maxb) maxb=b
    }
    print "bin,count"
    for (i=0; i<=maxb; i++) {
      if (i==0) { s=0; e=50 }
      else      { s=51 + (i-1)*BIN; e=s+(BIN-1) }
      label = s "-" e
      cnt = (i in c) ? c[i] : 0
      print label "," cnt
    }
  }
' "$INPUT" > "$OUTCSV"

echo "Wrote $OUTCSV"
