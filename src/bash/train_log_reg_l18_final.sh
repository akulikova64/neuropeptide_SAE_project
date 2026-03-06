python ../python/train_log_reg_l18_final.py \
  --positives "/Volumes/T7 Shield/layer_18_embeddings" \
  --negatives "/Volumes/T7 Shield/layer_18_negative_embeddings" \
  --winners_csv "/Volumes/T7 Shield/layer_18_F1_scores/winning_thresholds_l18.csv" \
  --num_features 320 \
  --out_model "/Volumes/T7 Shield/layer_18_classifiers/final/logreg_allfolds_top320.joblib" \
  --permitted_pos "../../data/combined_l18_positive_final.fasta" \
  --permitted_neg "../../data/negative_l18_dataset_final.fasta" \
  --seed 9687254
