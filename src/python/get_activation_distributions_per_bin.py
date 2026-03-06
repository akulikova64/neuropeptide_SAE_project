import os
import pandas as pd

# this script uses the maxpooled data. It deletes all features with zero activations and bins the
# remaining activations/features by position bins. 

# === Configuration ===
input_path = "../../data/concept_groups/embeddings/group_1/max_pooled_training/negative"
output_csv = "../../data/concept_groups/stats/group_1/activation_distribution_bins_negative.csv"
bin_size = 25
max_position = 599

# === Collect all annotated data
combined_data = []

for filename in os.listdir(input_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_path, filename)
        df = pd.read_csv(file_path)

        # Remove rows with zero activation
        df = df[df["max_activation"] != 0]

        # Compute bin label
        df["bin"] = df["position"].apply(
            lambda pos: f"{(pos // bin_size) * bin_size}-{(pos // bin_size) * bin_size + bin_size - 1}"
        )

        # Extract ID from filename (everything before first "_")
        seq_id = filename.split("_")[0]
        df["ID"] = seq_id

        combined_data.append(df)

# === Combine and save
final_df = pd.concat(combined_data, ignore_index=True)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
final_df.to_csv(output_csv, index=False)

print(f"✅ Combined activation bin data saved to: {output_csv}")