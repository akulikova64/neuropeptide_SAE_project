command python ../python/neuropeptide_classifier.py \
  --fasta "../../data/amphibian_analysis/amphibians_filter_out_enzymes_5/amphibians_deduplicated.fasta" \
  --model "/Volumes/T7 Shield/layer_18_classifiers/final/logreg_allfolds_top320.joblib" \
  --ranking "/Volumes/T7 Shield/layer_18_F1_scores/winning_thresholds_l18.csv" \
  --top_n 320 \
  --out "../../data/amphibian_analysis/amphibians_layer18_logreg_top320_predictions.csv"