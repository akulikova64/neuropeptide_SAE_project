python ../python/train_log_reg_l18.py \
  --positives "/Volumes/T7 Shield/layer_18_embeddings" \
  --negatives "/Volumes/T7 Shield/layer_18_negative_embeddings" \
  --ranking   "/Volumes/T7 Shield/layer_18_ranking_files" \
  --models_outdir    "/Volumes/T7 Shield/layer_18_classifiers/models" \
  --val_results_outdir "/Volumes/T7 Shield/layer_18_classifiers/val_results" \
  --postive_folds "../../data/combined_datasets_graphpart_mmseqs.csv" \
  --negative_folds "../../data/negative_dataset_10_folds.csv" \
  --permitted_pos "../../data/combined_l18_positive_final.fasta" \
  --permitted_neg "../../data/negative_l18_dataset_final.fasta" \
  --seed 9687254







