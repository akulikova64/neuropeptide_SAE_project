import os
import torch
import pandas as pd
from glob import glob

# Define input and output paths
input_base_path = "../../data/concept_groups/embeddings/group_1/training"
output_base_path = "../../data/concept_groups/embeddings/group_1/max_pooled_training"

# Ensure output directories exist
os.makedirs(f"{output_base_path}/positive", exist_ok=True)
os.makedirs(f"{output_base_path}/negative", exist_ok=True)

# Function to process and save max-pooled features
def process_embeddings(input_dir, output_dir):
    embedding_files = glob(os.path.join(input_dir, "*.pt"))
    
    for file_path in embedding_files:
        features = torch.load(file_path)  # shape: (sequence_length, num_features)
        max_vals, max_indices = torch.max(features, dim=0)

        df = pd.DataFrame({
            "feature": list(range(features.shape[1])),
            "max_activation": max_vals.tolist(),
            "position": max_indices.tolist()
        })

        filename = os.path.splitext(os.path.basename(file_path))[0] + "_max_pooled.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

# Run for both positive and negative datasets
process_embeddings(f"{input_base_path}/positive", f"{output_base_path}/positive")
process_embeddings(f"{input_base_path}/negative", f"{output_base_path}/negative")