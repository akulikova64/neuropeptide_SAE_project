# paste this into command line to split fasta into chunks of 250:

mkdir -p "../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5/deeptmhmm_chunks" && \
seqkit split -s 250 \
  -O "../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5/deeptmhmm_chunks" \
  "../../data/fig_6_zebrafish_secretome/secretome_no_enzymes_deduplicated.fasta" && \
cd "../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5/deeptmhmm_chunks" && \
setopt null_glob && \
i=1; for f in *.fa *.fasta; do mv -- "$f" "$(printf 'chunk_%04d.fa' "$i")"; i=$((i+1)); done


# then run the following script: 
# python deeptmhmm_batch_submit.py "../../data/fig_6_zebrafish_secretome/secretome_filter_out_enzymes_5"