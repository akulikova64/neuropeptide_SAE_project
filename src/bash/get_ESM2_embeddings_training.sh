#python ../python/get_ESM2_embeddings_training.py \
#  --fasta ../../data/combined_l18_positive_final.fasta \
#  --out-dir "/Volumes/T7 Shield/ESM2_embeddings_training/positive" \
#  --batch-size 4 \
#  --num-threads 4


python ../python/get_ESM2_embeddings_training.py \
  --fasta ../../data/negative_l18_dataset_final.fasta \
  --out-dir "/Volumes/T7 Shield/ESM2_embeddings_training/negative" \
  --batch-size 4 \
  --num-threads 4
