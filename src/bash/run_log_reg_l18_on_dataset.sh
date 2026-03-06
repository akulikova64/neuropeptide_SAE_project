python ../python/run_log_reg_l18_on_dataset.py \
  --model "/Volumes/T7 Shield/layer_18_classifiers/final/logreg_allfolds_top320.joblib" \
  --ranking "/Volumes/T7 Shield/layer_18_F1_scores/winning_thresholds_l18.csv" \
  --embeds "/Volumes/T7 Shield/layer_18_secretome_zebrafish" \
  --out "../../data/fig_6_zebrafish_secretome/layer_18_secretome_zebrafish_log_reg_320_predictions.csv" \
  --top_n 320



