import subprocess

# -----------------------
# File paths
# -----------------------

original_fasta = "../../data/fig_6_zebrafish_secretome/secretome_no_enzymes_deduplicated.fasta"

signalp_fasta  = "../../data/fig_6_zebrafish_secretome/secretome_no_TMRs_signalP_filtered.fasta"

cdhit_output   = "../../data/fig_6_zebrafish_secretome/secretome_final_cdhit_95.fasta"



# -----------------------
# Helper: count FASTA entries
# -----------------------
def count_fasta_entries(fasta_file):
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

# -----------------------
# Count before CD-HIT
# -----------------------
n_original = count_fasta_entries(original_fasta)
n_signalp  = count_fasta_entries(signalp_fasta)

# -----------------------
# Run CD-HIT
# -----------------------
cdhit_cmd = [
    "cd-hit",
    "-i", signalp_fasta,
    "-o", cdhit_output,
    "-c", "0.95",    # sequence identity threshold (change if needed)
    "-n", "5",      # word length for 0.9
    "-d", "0",      # keep full headers
]

print("Running CD-HIT...")
subprocess.run(cdhit_cmd, check=True)

# -----------------------
# Count after CD-HIT
# -----------------------
n_cdhit = count_fasta_entries(cdhit_output)

# -----------------------
# Report summary
# -----------------------
print("\n=== Secretome filtering summary ===")
print(f"Original sequences:           {n_original}")
print(f"After SignalP filtering:      {n_signalp}")
print(f"After CD-HIT clustering:      {n_cdhit}")
print(f"Removed by SignalP:           {n_original - n_signalp}")
print(f"Removed by CD-HIT:            {n_signalp - n_cdhit}")
print(f"Total removed overall:        {n_original - n_cdhit}")
