import os
import torch
import pandas as pd
from collections import defaultdict

# ============ Configuration ============
input_base_path = "../../data/concept_groups/embeddings/group_1/training/positive"
output_csv = "../../data/concept_groups/stats/group_1/summary_activation_bins_positive.csv"
bin_size = 25
max_position = 599  # Maximum position across all sequences

# ============ Initialize bins ============
# Create a list of bins with integer-based labels
bin_labels = [f"{start}-{start + bin_size - 1}" for start in range(0, max_position + 1, bin_size)]

# Ensure dictionaries include all bins with zero counts initially
zero_count_bins = defaultdict(int, {label: 0 for label in bin_labels})
non_zero_count_bins = defaultdict(int, {label: 0 for label in bin_labels})
sequence_counts = defaultdict(int, {bin_label: 0 for bin_label in bin_labels})

# ============ Process each .pt embedding file ============
for i, filename in enumerate(sorted(os.listdir(input_base_path)), start=1):
    if filename.endswith(".pt"):
        filepath = os.path.join(input_base_path, filename)
        features = torch.load(filepath)  # shape: (seq_len, feature_dim)
        seq_len, feature_dim = features.shape

        # Keep track of which bins this sequence touches
        bins_touched = set()

        for pos in range(seq_len):
            bin_start = (pos // bin_size) * bin_size
            bin_label = f"{bin_start}-{bin_start + bin_size - 1}"
            bins_touched.add(bin_label)

            for feat in range(feature_dim):
                value = features[pos, feat].item()
                if value == 0.0:
                    zero_count_bins[bin_label] += 1
                else:
                    non_zero_count_bins[bin_label] += 1

        # After going through all positions, updating sequence counts
        for bin_label in bins_touched:
            sequence_counts[bin_label] += 1

        print(f"✅ [{i}] Processed: {filename} (Length: {seq_len}, Bins touched: {len(bins_touched)})")

# ============ Sort bins by numeric start position ============
# Extract start value from each bin label to sort numerically
sorted_bins = sorted(bin_labels, key=lambda x: int(x.split('-')[0]))

# ============ Save summary CSV ============
df = pd.DataFrame({
    "position_bin": sorted_bins,
    "zero_count": [zero_count_bins[bin] for bin in sorted_bins],
    "non_zero_count": [non_zero_count_bins[bin] for bin in sorted_bins],
    "sequence_count": [sequence_counts[bin] for bin in sorted_bins]
})

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print("✅ Summary saved to:", output_csv)