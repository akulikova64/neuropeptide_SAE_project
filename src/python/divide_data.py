from Bio import SeqIO
import random
import os

# this program will first shuffle sequences in the positive and negative groups, 
# then it will divide the sequences into training and test datsets 
# (80% of sequeces for training and 20% of sequences for test)

#=========== input_paths ==============

positive_data_path = "../../data/concept_groups/sequences/group_1/group_1_positive.fasta"
negative_data_path = "../../data/concept_groups/sequences/group_1/group_1_negative.fasta"

#=========== output_paths ==============

positive_training = "../../data/concept_groups/sequences/group_1/training/positive.fasta"
negative_training = "../../data/concept_groups/sequences/group_1/training/negative.fasta"
positive_test = "../../data/concept_groups/sequences/group_1/test/positive.fasta"
negative_test = "../../data/concept_groups/sequences/group_1/test/negative.fasta"

# =========== Create output directories if they don't exist ==============
os.makedirs(os.path.dirname(positive_training), exist_ok=True)
os.makedirs(os.path.dirname(positive_test), exist_ok=True)

# =========== Read sequences ==============
positive_records = list(SeqIO.parse(positive_data_path, "fasta"))
negative_records = list(SeqIO.parse(negative_data_path, "fasta"))

assert len(positive_records) == len(negative_records), "Positive and negative sets must be the same length for paired shuffling."

# =========== Shuffle in parallel ==============
combined = list(zip(positive_records, negative_records))
random.shuffle(combined)
positive_records_shuffled, negative_records_shuffled = zip(*combined)

# =========== Split ==============
split_index = int(0.8 * len(positive_records))

pos_train = positive_records_shuffled[:split_index]
pos_test = positive_records_shuffled[split_index:]

neg_train = negative_records_shuffled[:split_index]
neg_test = negative_records_shuffled[split_index:]

# =========== Write to FASTA files ==============
SeqIO.write(pos_train, positive_training, "fasta")
SeqIO.write(pos_test, positive_test, "fasta")
SeqIO.write(neg_train, negative_training, "fasta")
SeqIO.write(neg_test, negative_test, "fasta")

# =========== Print counts ==============
print("✅ Done! Training and test sets written.")
print(f"Positive Training Sequences: {len(pos_train)}")
print(f"Negative Training Sequences: {len(neg_train)}")
print(f"Positive Test Sequences:     {len(pos_test)}")
print(f" Negative Test Sequences:     {len(neg_test)}")

