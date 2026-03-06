import os
import torch
import numpy as np

positive_dir = "../../data/concept_groups/embeddings/group_1/training/positive"
pos_files = [f for f in os.listdir(positive_dir) if f.endswith(".pt")]
num_pos = len(pos_files)

# load first to get the number of features
first = torch.load(os.path.join(positive_dir, pos_files[0]))
_, feature_dim = first.shape

# tally how many times each feature is non-zero in a given file
count_presence = np.zeros(feature_dim, dtype=int)
for fname in pos_files:
    data = torch.load(os.path.join(positive_dir, fname))   # (seq_len, feature_dim)
    seq_len = data.size(0)
    # count non-zeros per feature
    nonzero_counts = torch.count_nonzero(data, dim=0).numpy()
    # flag presence if at least one non-zero position
    count_presence += (nonzero_counts > 0).astype(int)

# now inspect
max_presence = count_presence.max()
print(f"Highest presence count = {max_presence}/{num_pos}")
if max_presence == num_pos:
    print("✅ At least one feature is non-zero in EVERY sequence!")
    feats = np.where(count_presence == num_pos)[0]
    print("Features that reach full coverage:", feats)
else:
    print("❌ No feature is non-zero in all sequences.")
    # show the top 10 features by presence count
    top10 = np.argsort(-count_presence)[:10]
    print("Top 10 by presence count:")
    for f in top10:
        print(f"  feature {f}: {count_presence[f]}/{num_pos}")