import os
import pandas as pd

def calculate_nonzero_proportion(folder_path):
    total_activations = 0
    nonzero_activations = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            activations = df["max_activation"]
            total_activations += len(activations)
            nonzero_activations += (activations != 0).sum()

    proportion = nonzero_activations / total_activations if total_activations > 0 else 0
    return proportion, total_activations, nonzero_activations


# === Folder paths ===
positive_path = "../../data/concept_groups/embeddings/group_1/max_pooled_training/positive"
negative_path = "../../data/concept_groups/embeddings/group_1/max_pooled_training/negative"

# === Calculate proportions ===
pos_prop, pos_total, pos_nonzero = calculate_nonzero_proportion(positive_path)
neg_prop, neg_total, neg_nonzero = calculate_nonzero_proportion(negative_path)

# === Output ===
print(f"✅ Positive Set:")
print(f"   Total activations: {pos_total}")
print(f"   Non-zero activations: {pos_nonzero}")
print(f"   Proportion non-zero: {pos_prop:.4f}\n")

print(f"✅ Negative Set:")
print(f"   Total activations: {neg_total}")
print(f"   Non-zero activations: {neg_nonzero}")
print(f"   Proportion non-zero: {neg_prop:.4f}")